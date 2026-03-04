import os
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = [
    "denormalize",
    "normalize",
    "tensor2pil",
    "pil2tensor",
    "save_image",
    "concat_inpainting_results",
    "generate_irregular_mask",
    "mask_image",
    "adjust_dynamic_range"
]


def denormalize(
    img: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> torch.Tensor:
    mean = torch.tensor(mean, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, 3, 1, 1)
    return img * std + mean


def normalize(
    img: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> torch.Tensor:
    mean = torch.tensor(mean, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def adjust_dynamic_range(
    img: torch.Tensor,
    drange_in: Tuple[float, float],
    drange_out: Tuple[float, float]
) -> torch.Tensor:
    scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
    bias = drange_out[0] - drange_in[0] * scale
    return img * scale + bias


def tensor2pil(img_tensor: torch.Tensor) -> Image.Image:
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img_np = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def pil2tensor(
    img_pil: Image.Image,
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    if normalize:
        img_tensor = adjust_dynamic_range(img_tensor, (0.0, 1.0), (-1.0, 1.0))
    if device is not None:
        img_tensor = img_tensor.to(device)
    return img_tensor


def save_image(
    img_tensor: torch.Tensor,
    save_path: str,
    denormalize_img: bool = True
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if denormalize_img:
        img_tensor = denormalize(img_tensor)
    img_pil = tensor2pil(img_tensor)
    img_pil.save(save_path)


def concat_inpainting_results(
    damaged_img: torch.Tensor,
    mask: torch.Tensor,
    init_img: torch.Tensor,
    pred_img: torch.Tensor,
    gt_img: torch.Tensor,
    nrow: Optional[int] = None
) -> torch.Tensor:
    B, C, H, W = damaged_img.shape
    nrow = nrow if nrow is not None else B

    # Denormalize all images to [0, 1]
    damaged = denormalize(damaged_img)
    init = denormalize(init_img)
    pred = denormalize(pred_img)
    gt = denormalize(gt_img)

    # Convert mask to 3-channel for visualization
    mask_vis = mask.repeat(1, 3, 1, 1)

    # Concatenate per sample
    concat_list = []
    for i in range(B):
        sample_concat = torch.cat([
            damaged[i],
            mask_vis[i],
            init[i],
            pred[i],
            gt[i]
        ], dim=2)
        concat_list.append(sample_concat)

    # Concatenate batch
    if nrow == 1:
        final_concat = torch.cat(concat_list, dim=1)
    else:
        rows = []
        for i in range(0, B, nrow):
            row = torch.cat(concat_list[i:i+nrow], dim=2)
            rows.append(row)
        final_concat = torch.cat(rows, dim=1)

    return final_concat


def generate_irregular_mask(
    img_size: int = 256,
    min_holes: int = 1,
    max_holes: int = 8,
    min_mask_ratio: float = 0.05,
    max_mask_ratio: float = 0.6,
    max_vertex: int = 10,
    max_angle: float = np.pi / 5,
    max_length: int = 50,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    mask = np.zeros((img_size, img_size), dtype=np.float32)
    mask_ratio = np.random.uniform(min_mask_ratio, max_mask_ratio)
    num_holes = np.random.randint(min_holes, max_holes + 1)

    for _ in range(num_holes):
        if np.sum(mask) / (img_size ** 2) >= mask_ratio:
            break

        vertex = np.random.randint(3, max_vertex + 1)
        start_x = np.random.randint(0, img_size)
        start_y = np.random.randint(0, img_size)
        angle = np.random.uniform(0, np.pi * 2)

        points = []
        for _ in range(vertex):
            r = np.random.randint(0, max_length + 1)
            angle += np.random.uniform(-max_angle, max_angle)
            x = np.clip(start_x + r * np.cos(angle), 0, img_size - 1)
            y = np.clip(start_y + r * np.sin(angle), 0, img_size - 1)
            points.append((int(x), int(y)))

        points = np.array(points, dtype=np.int32)
        try:
            import cv2
            cv2.fillPoly(mask, [points], 1.0)
        except ImportError:
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch
            y, x = np.mgrid[:img_size, :img_size]
            points = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
            path = Path(points)
            grid = path.contains_points(points)
            mask += grid.reshape(img_size, img_size).astype(np.float32)

    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    if device is not None:
        mask_tensor = mask_tensor.to(device)
    return mask_tensor


def mask_image(
    img: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = -1.0
) -> torch.Tensor:
    return img * (1 - mask) + mask_value * mask


# Runtime verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test mask generation
    test_mask = generate_irregular_mask(img_size=256, device=device)
    print(f"Generated mask shape: {test_mask.shape}, unique values: {torch.unique(test_mask)}")

    # Test image normalization
    test_img = torch.randn(2, 3, 256, 256).to(device)
    test_img_norm = normalize(test_img)
    test_img_denorm = denormalize(test_img_norm)
    print(f"Normalized image range: [{test_img_norm.min():.2f}, {test_img_norm.max():.2f}]")
    print(f"Denormalized image range: [{test_img_denorm.min():.2f}, {test_img_denorm.max():.2f}]")

    # Test result concatenation
    test_damaged = torch.randn(4, 3, 256, 256).to(device)
    test_init = torch.randn(4, 3, 256, 256).to(device)
    test_pred = torch.randn(4, 3, 256, 256).to(device)
    test_gt = torch.randn(4, 3, 256, 256).to(device)
    test_mask_batch = torch.zeros(4, 1, 256, 256).to(device)
    test_mask_batch[:, :, 100:150, 100:150] = 1.0

    concat_result = concat_inpainting_results(test_damaged, test_mask_batch, test_init, test_pred, test_gt, nrow=2)
    print(f"Concatenated result shape: {concat_result.shape}")

    print("Image utils test completed.")
