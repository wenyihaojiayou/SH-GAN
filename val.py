import os
import argparse
import torch
import yaml
from tqdm import tqdm

from models.shgan import SHGAN
from datasets.inpainting_dataset import InpaintingDataset
from metrics.psnr_ssim import PSNRSSIMMetrics
from metrics.lpips_metric import LPIPS
from metrics.fid_metric import FID
from utils.logger import get_logger
from utils.image_utils import concat_inpainting_results, save_image
from utils.distributed import *

def parse_args():
    parser = argparse.ArgumentParser(description="SH-GAN Validation")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name, override config")
    parser.add_argument("--save_results", action="store_true", help="Save validation results")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset

    # Device setup
    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = is_main_process()

    # Logger
    logger = get_logger(
        log_dir=config["paths"]["log_dir"],
        exp_name=f"{config['project_name']}_val",
        use_tensorboard=False,
        is_main_process=is_main
    )

    # Dataset
    dataset_name = config["dataset"]["name"]
    val_dataset = InpaintingDataset(
        dataset_name=dataset_name,
        split="val",
        data_root=os.path.join(config["paths"]["data_root"], dataset_name),
        mask_root=config["paths"]["mask_root"],
        img_size=config["dataset"]["img_size"],
        use_online_mask=False,
        return_init_img=False
    )
    val_loader, _ = prepare_distributed_dataloader(
        val_dataset,
        batch_size=config["evaluation"]["test_batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["dataloader"]["num_workers"],
        pin_memory=config["dataset"]["dataloader"]["pin_memory"],
        drop_last=False
    )
    if is_main:
        logger.info(f"Validation dataset loaded: {len(val_dataset)} samples")

    # Model initialization
    model = SHGAN(
        block_size=config["model"]["shtm"]["block_size"],
        search_window=config["model"]["shtm"]["search_window"],
        topk_blocks=config["model"]["shtm"]["topk_blocks"],
        topk_rows=config["model"]["shtm"]["topk_rows"],
        tau_valid_ratio=config["model"]["shtm"]["tau_valid_ratio"],
        freeze_shtm=True,
        gen_in_channels=config["model"]["generator"]["in_channels"],
        gen_base_channels=config["model"]["generator"]["base_channels"],
        num_heads=config["model"]["hmam"]["num_heads"],
        d_head=config["model"]["hmam"]["d_head"],
        mem_dim=config["model"]["hmam"]["mem_dim"],
        disc_in_channels=config["model"]["discriminator"]["in_channels"],
        disc_base_channels=config["model"]["discriminator"]["base_channels"],
        crop_size=config["model"]["discriminator"]["crop_size"]
    ).to(device)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if is_main:
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logger.error(f"Checkpoint {args.checkpoint} not found!")
        return
    model.eval()

    # Metrics initialization
    psnr_ssim_metrics = PSNRSSIMMetrics().to(device)
    lpips_metrics = LPIPS().to(device)
    fid_calculator = FID(device=device)

    masked_eval = False

    fid_max_samples = config["evaluation"]["fid"].get("max_samples", 1000)

    # Validation loop
    if is_main:
        logger.info(f"Start validation on {dataset_name} dataset, masked_eval={masked_eval}")
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_pred_imgs = []
    all_gt_imgs = []

    pbar = tqdm(val_loader, disable=not is_main)
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            damaged_img, mask, gt_img, _ = batch
            damaged_img = damaged_img.to(device)
            mask = mask.to(device)
            gt_img = gt_img.to(device)

            # Model forward
            pred_img, _, init_img = model(damaged_img, mask)

            # Metrics calculation
            eval_mask = mask if masked_eval else None
            psnr, ssim = psnr_ssim_metrics(pred_img, gt_img, mask=eval_mask, reduction="none")
            lpips = lpips_metrics(pred_img, gt_img, mask=eval_mask, reduction="none")

            all_psnr.append(psnr.cpu())
            all_ssim.append(ssim.cpu())
            all_lpips.append(lpips.cpu())
            all_pred_imgs.append(pred_img.cpu())
            all_gt_imgs.append(gt_img.cpu())

            # Update progress bar
            pbar.set_postfix({
                "PSNR": f"{psnr.mean().item():.2f}",
                "SSIM": f"{ssim.mean().item():.4f}",
                "LPIPS": f"{lpips.mean().item():.4f}"
            })

            # Save results
            if args.save_results and is_main:
                save_dir = os.path.join(config["paths"]["result_dir"], f"val_{dataset_name}")
                concat_img = concat_inpainting_results(damaged_img, mask, init_img, pred_img, gt_img, nrow=1)
                save_image(concat_img, os.path.join(save_dir, f"sample_{i}.png"))

    # Gather results from all processes
    all_psnr = torch.cat(all_psnr, dim=0)
    all_ssim = torch.cat(all_ssim, dim=0)
    all_lpips = torch.cat(all_lpips, dim=0)

    if is_distributed():
        all_psnr = all_gather_tensor(all_psnr.to(device))
        all_ssim = all_gather_tensor(all_ssim.to(device))
        all_lpips = all_gather_tensor(all_lpips.to(device))

    # Calculate average metrics
    avg_psnr = all_psnr.mean().item()
    avg_ssim = all_ssim.mean().item()
    avg_lpips = all_lpips.mean().item()

    # Calculate FID
    if is_main:
        logger.info("Calculating FID score...")
        # Create dummy dataloaders for FID
        from torch.utils.data import TensorDataset, DataLoader
        all_pred_imgs = torch.cat(all_pred_imgs, dim=0)
        all_gt_imgs = torch.cat(all_gt_imgs, dim=0)

        pred_dataset = TensorDataset(all_pred_imgs)
        gt_dataset = TensorDataset(all_gt_imgs)

        pred_loader = DataLoader(pred_dataset, batch_size=config["evaluation"]["fid"]["batch_size"], shuffle=False)
        gt_loader = DataLoader(gt_dataset, batch_size=config["evaluation"]["fid"]["batch_size"], shuffle=False)

        fid_score = fid_calculator.compute_fid(gt_loader, pred_loader, max_samples=fid_max_samples)
        logger.info(f"FID calculation completed")

        # Log final results
        logger.info("="*50 + " Validation Results " + "="*50)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"PSNR: {avg_psnr:.4f} dB")
        logger.info(f"SSIM: {avg_ssim:.4f}")
        logger.info(f"LPIPS: {avg_lpips:.6f}")
        logger.info(f"FID: {fid_score:.4f}")
        logger.info("="*112)

    cleanup_distributed()

if __name__ == "__main__":
    main()