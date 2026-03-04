import os
import random
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class IrregularMaskGenerator:
    """
    Irregular mask generator aligned with the paper's mask configuration
    """
    def __init__(
        self,
        img_size: int = 256,
        min_holes: int = 1,
        max_holes: int = 8,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.6,
        max_vertex: int = 10,
        max_angle: float = np.pi / 5,
        max_length: int = 50
    ):
        self.img_size = img_size
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.max_vertex = max_vertex
        self.max_angle = max_angle
        self.max_length = max_length

    def generate_mask(self) -> np.ndarray:
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        num_holes = random.randint(self.min_holes, self.max_holes)

        for _ in range(num_holes):
            if np.sum(mask) / (self.img_size ** 2) >= mask_ratio:
                break

            vertex = random.randint(3, self.max_vertex)
            start_x = random.randint(0, self.img_size)
            start_y = random.randint(0, self.img_size)
            angle = random.uniform(0, np.pi * 2)

            points = []
            for _ in range(vertex):
                r = random.randint(0, self.max_length)
                angle += random.uniform(-self.max_angle, self.max_angle)
                x = np.clip(start_x + r * np.cos(angle), 0, self.img_size - 1)
                y = np.clip(start_y + r * np.sin(angle), 0, self.img_size - 1)
                points.append((int(x), int(y)))

            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1.0)

        return mask


class InpaintingDataset(Dataset):
    """
    Image Inpainting Dataset, strictly aligned with the dataset configuration in the paper
    """


    def __init__(
        self,
        dataset_name: str,
        split: str,
        data_root: str,
        mask_root: Optional[str] = None,
        img_size: int = 256,
        use_online_mask: bool = True,
        return_init_img: bool = False,
        init_img_root: Optional[str] = None
    ):
        super().__init__()
        if dataset_name not in self.DATASET_CONFIG:
            raise ValueError(f"Dataset {dataset_name} not supported, available: {list(self.DATASET_CONFIG.keys())}")
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split {split} not supported, available: train/val/test")

        self.dataset_name = dataset_name
        self.split = split
        self.data_root = data_root
        self.mask_root = mask_root
        self.img_size = img_size
        self.use_online_mask = use_online_mask
        self.return_init_img = return_init_img
        self.init_img_root = init_img_root

        # Image transform, aligned with paper's 256x256 resolution requirement
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        # Load image paths
        self.img_paths = self._load_image_paths()

        # Initialize mask generator or load pre-generated masks
        if self.use_online_mask:
            self.mask_generator = IrregularMaskGenerator(img_size=img_size)
        else:
            self.mask_paths = self._load_mask_paths()

        # Load precomputed SHTM initialization images if enabled
        if self.return_init_img:
            if not init_img_root:
                raise ValueError("init_img_root must be provided when return_init_img is True")
            self.init_img_paths = self._load_init_img_paths()

    def _load_image_paths(self) -> List[str]:
        """Load image paths according to the dataset split in the paper"""
        img_dir = os.path.join(self.data_root, self.split)
        if not os.path.exists(img_dir):
            img_dir = self.data_root

        all_imgs = []
        for ext in ["jpg", "jpeg", "png", "bmp"]:
            all_imgs.extend(sorted(glob.glob(os.path.join(img_dir, f"*.{ext}"))))
            all_imgs.extend(sorted(glob.glob(os.path.join(img_dir, f"*.{ext.upper()}"))))

        # Align with the exact sample count in the paper
        expected_count = self.DATASET_CONFIG[self.dataset_name][self.split]
        if len(all_imgs) > expected_count:
            if self.split == "train":
                all_imgs = all_imgs[:expected_count]
            else:
                all_imgs = all_imgs[-expected_count:]
        elif len(all_imgs) < expected_count:
            raise ValueError(f"Expected {expected_count} images for {self.dataset_name} {self.split}, but found {len(all_imgs)}")

        return all_imgs

    def _load_mask_paths(self) -> List[str]:
        """Load pre-generated irregular mask paths (20,000 masks as described in the paper)"""
        mask_paths = []
        for ext in ["png", "jpg", "bmp"]:
            mask_paths.extend(sorted(glob.glob(os.path.join(self.mask_root, f"*.{ext}"))))
            mask_paths.extend(sorted(glob.glob(os.path.join(self.mask_root, f"*.{ext.upper()}"))))
        if len(mask_paths) == 0:
            raise ValueError(f"No mask files found in {self.mask_root}")
        return mask_paths

    def _load_init_img_paths(self) -> List[str]:
        """Load precomputed SHTM initialization image paths"""
        init_paths = []
        for ext in ["png", "jpg", "npy"]:
            init_paths.extend(sorted(glob.glob(os.path.join(self.init_img_root, f"*.{ext}"))))
            init_paths.extend(sorted(glob.glob(os.path.join(self.init_img_root, f"*.{ext.upper()}"))))
        if len(init_paths) != len(self.img_paths):
            raise ValueError(f"Init image count {len(init_paths)} does not match image count {len(self.img_paths)}")
        return init_paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Load ground truth image
        img_path = self.img_paths[idx]
        img_pil = Image.open(img_path).convert("RGB")
        gt_img = self.img_transform(img_pil)

        # Load or generate mask (1 = damaged region, 0 = known region, aligned with paper definition)
        if self.use_online_mask:
            mask_np = self.mask_generator.generate_mask()
            mask = torch.from_numpy(mask_np).unsqueeze(0).float()
        else:
            mask_idx = random.randint(0, len(self.mask_paths) - 1) if self.split == "train" else idx % len(self.mask_paths)
            mask_pil = Image.open(self.mask_paths[mask_idx]).convert("L")
            mask = self.mask_transform(mask_pil)
            mask = (mask > 0.5).float()

        # Generate damaged image
        damaged_img = gt_img * (1 - mask)

        # Load precomputed SHTM initialization image if enabled
        init_img = None
        if self.return_init_img:
            init_path = self.init_img_paths[idx]
            if init_path.endswith(".npy"):
                init_np = np.load(init_path)
                init_img = torch.from_numpy(init_np).float()
            else:
                init_pil = Image.open(init_path).convert("RGB")
                init_img = self.img_transform(init_pil)

        return damaged_img, mask, gt_img, init_img


# Import here to avoid top-level dependency for users who don't use online mask generation
try:
    import cv2
    import glob
except ImportError:
    pass


# Runtime verification
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Test dataset initialization
    test_dataset = InpaintingDataset(
        dataset_name="Facade",
        split="test",
        data_root="./data/facade",
        use_online_mask=True,
        img_size=256
    )
    print(f"Dataset length: {len(test_dataset)}")

    # Test single sample
    damaged_img, mask, gt_img, init_img = test_dataset[0]
    print(f"Damaged image shape: {damaged_img.shape}, range: [{damaged_img.min():.2f}, {damaged_img.max():.2f}]")
    print(f"Mask shape: {mask.shape}, unique values: {torch.unique(mask)}")
    print(f"GT image shape: {gt_img.shape}, range: [{gt_img.min():.2f}, {gt_img.max():.2f}]")

    # Test dataloader
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(test_loader))
    damaged_batch, mask_batch, gt_batch, init_batch = batch
    print(f"Batch damaged shape: {damaged_batch.shape}")
    print(f"Batch mask shape: {mask_batch.shape}")
    print(f"Batch GT shape: {gt_batch.shape}")
    print("Dataset module test completed.")
