import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import functional as TF
from typing import List, Tuple, Optional
import numpy as np
from scipy import linalg

__all__ = ["FIDFeatureExtractor", "FID"]


class FIDFeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        eval_mode: bool = True
    ):
        super().__init__()
        inception = models.inception_v3(pretrained=pretrained, transform_input=False)
        self.block1 = inception.Conv2d_1a_3x3
        self.block2 = inception.Conv2d_2a_3x3
        self.block3 = inception.Conv2d_2b_3x3
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.block4 = inception.Conv2d_3b_1x1
        self.block5 = inception.Conv2d_4a_3x3
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.mixed_5b = inception.Mixed_5b
        self.mixed_5c = inception.Mixed_5c
        self.mixed_5d = inception.Mixed_5d
        self.mixed_6a = inception.Mixed_6a
        self.mixed_6b = inception.Mixed_6b
        self.mixed_6c = inception.Mixed_6c
        self.mixed_6d = inception.Mixed_6d
        self.mixed_6e = inception.Mixed_6e
        self.mixed_7a = inception.Mixed_7a
        self.mixed_7b = inception.Mixed_7b
        self.mixed_7c = inception.Mixed_7c
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Input normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        if eval_mode:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input preprocessing
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # Forward pass through InceptionV3
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool1(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.maxpool2(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.mixed_5d(x)
        x = self.mixed_6a(x)
        x = self.mixed_6b(x)
        x = self.mixed_6c(x)
        x = self.mixed_6d(x)
        x = self.mixed_6e(x)
        x = self.mixed_7a(x)
        x = self.mixed_7b(x)
        x = self.mixed_7c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class FID:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        eps: float = 1e-6
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FIDFeatureExtractor().to(self.device)
        self.eps = eps

    @torch.no_grad()
    def extract_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        features_list = []
        sample_count = 0

        for batch in dataloader:
            if len(batch) == 4:
                damaged, mask, gt, init = batch
                img = gt
            else:
                img = batch[0]

            img = img.to(self.device)
            features = self.feature_extractor(img)
            features_list.append(features.cpu().numpy())

            sample_count += img.shape[0]
            if max_samples is not None and sample_count >= max_samples:
                break

        features = np.concatenate(features_list, axis=0)
        if max_samples is not None:
            features = features[:max_samples]
        return features

    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray
    ) -> float:
        mu1 = mu1.astype(np.float64)
        mu2 = mu2.astype(np.float64)
        sigma1 = sigma1.astype(np.float64)
        sigma2 = sigma2.astype(np.float64)

        # Mean difference squared
        diff = mu1 - mu2
        mean_diff_sq = np.sum(diff ** 2)

        # Covariance sqrt product
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * self.eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Remove imaginary parts from numerical error
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Trace term
        trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)

        # Final FID score
        fid = mean_diff_sq + trace_term
        return float(fid)

    @torch.no_grad()
    def compute_fid(
        self,
        real_dataloader: torch.utils.data.DataLoader,
        fake_dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None
    ) -> float:
        # Extract features for real and fake images
        real_features = self.extract_features(real_dataloader, max_samples)
        fake_features = self.extract_features(fake_dataloader, max_samples)

        # Compute statistics
        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_fake, sigma_fake = self.compute_statistics(fake_features)

        # Calculate FID
        fid_score = self.calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        return fid_score


# Runtime verification
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_calculator = FID(device=device)

    # Create test dataset
    test_real = torch.randn(50, 3, 256, 256)
    test_fake = torch.randn(50, 3, 256, 256)

    real_dataset = TensorDataset(test_real)
    fake_dataset = TensorDataset(test_fake)

    real_loader = DataLoader(real_dataset, batch_size=10, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=10, shuffle=False)

    fid_score = fid_calculator.compute_fid(real_loader, fake_loader)
    print(f"Test FID Score: {fid_score:.4f}")
    print("FID metric test completed.")
