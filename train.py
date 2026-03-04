import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from models.shgan import SHGAN
from datasets.inpainting_dataset import InpaintingDataset
from losses.generator_loss import GeneratorLoss
from losses.discriminator_loss import DiscriminatorLoss
from metrics.psnr_ssim import PSNRSSIMMetrics
from utils.logger import get_logger
from utils.image_utils import concat_inpainting_results, save_image
from utils.distributed import *

def parse_args():
    parser = argparse.ArgumentParser(description="SH-GAN Training")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Distributed initialization
    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = is_main_process()

    # Random seed setting
    seed = config["seed"]
    torch.manual_seed(seed)
    if is_main:
        np.random.seed(seed)
        random.seed(seed)
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = config["deterministic"]

    # Logger initialization
    logger = get_logger(
        log_dir=config["paths"]["log_dir"],
        exp_name=config["project_name"],
        use_tensorboard=config["logging"]["use_tensorboard"],
        is_main_process=is_main
    )
    if is_main:
        logger.log_config(config)
        logger.info(f"Distributed training: world size {world_size}, rank {rank}")

    # Dataset and dataloader
    dataset_name = config["dataset"]["name"]
    train_dataset = InpaintingDataset(
        dataset_name=dataset_name,
        split="train",
        data_root=os.path.join(config["paths"]["data_root"], dataset_name),
        mask_root=config["paths"]["mask_root"],
        img_size=config["dataset"]["img_size"],
        use_online_mask=config["dataset"]["mask"]["use_online_generation"],
        return_init_img=False
    )
    train_loader, train_sampler = prepare_distributed_dataloader(
        train_dataset,
        batch_size=config["dataset"]["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["dataloader"]["num_workers"],
        pin_memory=config["dataset"]["dataloader"]["pin_memory"],
        drop_last=config["dataset"]["dataloader"]["drop_last"]
    )
    if is_main:
        logger.info(f"Train dataset loaded: {len(train_dataset)} samples")

    # Validation dataset
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
        # SHTM params
        block_size=config["model"]["shtm"]["block_size"],
        search_window=config["model"]["shtm"]["search_window"],
        topk_blocks=config["model"]["shtm"]["topk_blocks"],
        topk_rows=config["model"]["shtm"]["topk_rows"],
        tau_valid_ratio=config["model"]["shtm"]["tau_valid_ratio"],
        freeze_shtm=config["model"]["shtm"]["freeze_shtm"],
        # Generator params
        gen_in_channels=config["model"]["generator"]["in_channels"],
        gen_base_channels=config["model"]["generator"]["base_channels"],
        # HMAM params
        num_heads=config["model"]["hmam"]["num_heads"],
        d_head=config["model"]["hmam"]["d_head"],
        mem_dim=config["model"]["hmam"]["mem_dim"],
        # Discriminator params
        disc_in_channels=config["model"]["discriminator"]["in_channels"],
        disc_base_channels=config["model"]["discriminator"]["base_channels"],
        crop_size=config["model"]["discriminator"]["crop_size"]
    ).to(device)

    if not config["model"]["shtm"].get("trainable", False):
        for param in model.shtm.parameters():
            param.requires_grad = False

    # Distributed model wrap
    if is_distributed():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )
    model_without_ddp = model.module if is_distributed() else model

    # Optimizer and lr scheduler
    gen_optimizer = optim.Adam(
        model_without_ddp.get_generator_params(),
        lr=config["optimizer"]["generator"]["lr"],
        betas=config["optimizer"]["generator"]["betas"],
        weight_decay=config["optimizer"]["generator"]["weight_decay"]
    )
    disc_optimizer = optim.Adam(
        model_without_ddp.get_discriminator_params(),
        lr=config["optimizer"]["discriminator"]["lr"],
        betas=config["optimizer"]["discriminator"]["betas"],
        weight_decay=config["optimizer"]["discriminator"]["weight_decay"]
    )

    # LR scheduler
    warmup_steps = config["lr_scheduler"]["warmup_steps"]
    gen_scheduler = optim.lr_scheduler.StepLR(
        gen_optimizer,
        step_size=config["lr_scheduler"]["decay_steps"],
        gamma=config["lr_scheduler"]["decay_rate"]
    )
    disc_scheduler = optim.lr_scheduler.StepLR(
        disc_optimizer,
        step_size=config["lr_scheduler"]["decay_steps"],
        gamma=config["lr_scheduler"]["decay_rate"]
    )

    # Loss function initialization
    gen_criterion = GeneratorLoss(
        lambda_adv=config["loss"]["generator"]["lambda_adv"],
        lambda_rec=config["loss"]["generator"]["lambda_rec"],
        lambda_style=config["loss"]["generator"]["lambda_style"] * 0.7,
        lambda_color=config["loss"]["generator"]["lambda_color"] * 0.8
    ).to(device)
    disc_criterion = DiscriminatorLoss(
        lambda_gp=config["loss"]["discriminator"]["lambda_gp"],
        lambda_mix=config["loss"]["discriminator"]["lambda_mix"]
    ).to(device)

    # Evaluation metrics
    metrics = PSNRSSIMMetrics().to(device)

    # Resume training
    start_epoch = 0
    global_step = 0
    max_epochs = config["training"]["max_epochs"][dataset_name]
    if args.resume or config["training"]["resume_training"]:
        resume_path = config["training"]["resume_checkpoint"]
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            model_without_ddp.load_state_dict(checkpoint["model_state_dict"])
            gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
            disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
            gen_scheduler.load_state_dict(checkpoint["gen_scheduler_state_dict"])
            disc_scheduler.load_state_dict(checkpoint["disc_scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint["global_step"]
            if is_main:
                logger.info(f"Resume training from epoch {start_epoch}, step {global_step}")
        else:
            if is_main:
                logger.warn(f"Checkpoint {resume_path} not found, start from scratch")

    # AMP training
    scaler = GradScaler(enabled=config["training"]["fp16"])
    n_critic = config["loss"]["n_critic"]

    # Training loop
    if is_main:
        logger.info(f"Start training for {max_epochs} epochs")
    for epoch in range(start_epoch, max_epochs):
        if is_distributed():
            train_sampler.set_epoch(epoch)
        model.train()

        # Progress bar
        pbar = tqdm(train_loader, disable=not is_main)
        pbar.set_description(f"Epoch [{epoch+1}/{max_epochs}]")

        epoch_loss = {
            "gen_total": 0.0, "gen_adv": 0.0, "gen_recon": 0.0,
            "disc_total": 0.0, "disc_adv": 0.0, "disc_gp": 0.0, "disc_mix": 0.0,
            "train_psnr": 0.0
        }
        batch_count = 0

        for batch in pbar:
            damaged_img, mask, gt_img, _ = batch
            damaged_img = damaged_img.to(device)
            mask = mask.to(device)
            gt_img = gt_img.to(device)
            batch_size = damaged_img.shape[0]

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(n_critic):
                disc_optimizer.zero_grad()
                with autocast(enabled=config["training"]["fp16"]):
                    # Generator forward
                    pred_img, _, init_img = model(damaged_img, mask)
                    # Discriminator loss
                    disc_loss, disc_loss_log = disc_criterion(
                        model_without_ddp.discriminator, gt_img, pred_img, mask
                    )
                # Backward
                scaler.scale(disc_loss).backward()
                scaler.step(disc_optimizer)
                scaler.update()

            # ---------------------
            #  Train Generator
            # ---------------------
            gen_optimizer.zero_grad()
            with autocast(enabled=config["training"]["fp16"]):
                pred_img, _, init_img = model(damaged_img, mask)
                # Discriminator forward
                disc_outputs, _ = model_without_ddp.discriminate(pred_img, mask)
                disc_outputs = disc_outputs[:3]
                # Generator loss
                gen_loss, gen_loss_log = gen_criterion(pred_img, gt_img, mask, disc_outputs)
            # Backward
            scaler.scale(gen_loss).backward()
            # Gradient clipping
            if config["training"]["gradient_clip_val"] > 0:
                nn.utils.clip_grad_norm_(model_without_ddp.generator.parameters(), config["training"]["gradient_clip_val"])
            scaler.step(gen_optimizer)
            scaler.update()

            # LR warmup
            if global_step < warmup_steps:
                lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
                for pg in gen_optimizer.param_groups:
                    pg["lr"] = lr_scale * config["optimizer"]["generator"]["lr"]
                for pg in disc_optimizer.param_groups:
                    pg["lr"] = lr_scale * config["optimizer"]["discriminator"]["lr"]

            # Metrics calculation
            with torch.no_grad():
                psnr, _ = metrics(pred_img, gt_img, mask=None)

            # Update loss log
            for k in epoch_loss.keys():
                if k in gen_loss_log:
                    epoch_loss[k] += gen_loss_log[k] * batch_size
                elif k in disc_loss_log:
                    epoch_loss[k] += disc_loss_log[k] * batch_size
                elif k == "train_psnr":
                    epoch_loss[k] += psnr * batch_size
            batch_count += batch_size
            global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "Gen Loss": f"{gen_loss_log['total_loss']:.4f}",
                "Disc Loss": f"{disc_loss_log['total_loss']:.4f}",
                "PSNR": f"{psnr:.2f}",
                "Step": global_step
            })

            # Log to tensorboard
            if is_main and global_step % config["logging"]["log_interval"] == 0:
                for k, v in gen_loss_log.items():
                    logger.log_scalar(f"Train/Generator/{k}", v, global_step)
                for k, v in disc_loss_log.items():
                    logger.log_scalar(f"Train/Discriminator/{k}", v, global_step)
                logger.log_scalar(f"Train/PSNR", psnr, global_step)
                logger.log_scalar(f"Train/LR/Generator", gen_optimizer.param_groups[0]["lr"], global_step)
                logger.log_scalar(f"Train/LR/Discriminator", disc_optimizer.param_groups[0]["lr"], global_step)

        # Epoch average loss
        for k in epoch_loss.keys():
            epoch_loss[k] /= batch_count
        if is_main:
            logger.info(f"Epoch {epoch+1} completed. Average Gen Loss: {epoch_loss['gen_total']:.4f}, Average Disc Loss: {epoch_loss['disc_total']:.4f}, Average PSNR: {epoch_loss['train_psnr']:.2f}")

        # LR scheduler step
        if global_step >= warmup_steps:
            gen_scheduler.step()
            disc_scheduler.step()

        # Validation
        if is_main and (epoch + 1) % config["evaluation"]["val_interval"] == 0:
            model.eval()
            val_psnr_total = 0.0
            val_ssim_total = 0.0
            val_batch_count = 0
            val_pbar = tqdm(val_loader, desc="Validation")

            with torch.no_grad():
                for i, batch in enumerate(val_pbar):
                    damaged_img, mask, gt_img, _ = batch
                    damaged_img = damaged_img.to(device)
                    mask = mask.to(device)
                    gt_img = gt_img.to(device)

                    pred_img, _, init_img = model(damaged_img, mask)
                    psnr, ssim = metrics(pred_img, gt_img, mask=None)

                    val_psnr_total += psnr * damaged_img.shape[0]
                    val_ssim_total += ssim * damaged_img.shape[0]
                    val_batch_count += damaged_img.shape[0]
                    val_pbar.set_postfix({"PSNR": f"{psnr:.2f}", "SSIM": f"{ssim:.4f}"})

                    # Save visualization
                    if config["evaluation"]["save_val_results"] and i < 5:
                        concat_img = concat_inpainting_results(damaged_img, mask, init_img, pred_img, gt_img, nrow=1)
                        save_image(concat_img, os.path.join(config["paths"]["result_dir"], f"val/epoch_{epoch+1}/sample_{i}.png"))

            val_psnr = val_psnr_total / val_batch_count
            val_ssim = val_ssim_total / val_batch_count
            logger.info(f"Validation Epoch {epoch+1}: PSNR={val_psnr:.4f}, SSIM={val_ssim:.4f}")
            logger.log_scalar("Val/PSNR", val_psnr, epoch+1)
            logger.log_scalar("Val/SSIM", val_ssim, epoch+1)

        # Save checkpoint
        if is_main and (epoch + 1) % config["logging"]["save_checkpoint_interval"] == 0:
            checkpoint_dir = config["paths"]["checkpoint_dir"]
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"shgan_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model_without_ddp.state_dict(),
                "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                "disc_optimizer_state_dict": disc_optimizer.state_dict(),
                "gen_scheduler_state_dict": gen_scheduler.state_dict(),
                "disc_scheduler_state_dict": disc_scheduler.state_dict(),
                "config": config
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")


            checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")], key=lambda x: int(x.split("_")[2].split(".")[0]))
            if len(checkpoint_files) > config["logging"]["keep_checkpoint_max"]:
                for f in checkpoint_files[:-config["logging"]["keep_checkpoint_max"]]:
                    os.remove(os.path.join(checkpoint_dir, f))

    # Training completed
    if is_main:
        logger.info("Training completed!")
        final_checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], "shgan_final.pth")
        torch.save({
            "model_state_dict": model_without_ddp.state_dict(),
            "config": config
        }, final_checkpoint_path)
        logger.info(f"Final model saved to {final_checkpoint_path}")
        logger.close()

    cleanup_distributed()

if __name__ == "__main__":
    main()