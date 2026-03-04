import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = ["PSNR", "SSIM", "PSNRSSIMMetrics"]


def create_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device
) -> torch.Tensor:
    kernel_1d = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=device)
    kernel_1d = torch.exp(-kernel_1d ** 2 / (2 * sigma ** 2))
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.repeat(channels, 1, 1, 1)
    return kernel


class PSNR(nn.Module):
    def __init__(
        self,
        data_range: float = 1.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.data_range = data_range
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = (pred + 1.0) / 2.0 * self.data_range
        target = (target + 1.0) / 2.0 * self.data_range

        if mask is not None:
            mask = F.interpolate(mask, size=pred.shape[2:], mode="nearest")
            mse = torch.sum(mask * (pred - target) ** 2, dim=[1, 2, 3]) / (torch.sum(mask, dim=[1, 2, 3]) + self.eps)
        else:
            mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])

        psnr = 10 * torch.log10((self.data_range ** 2) / (mse + self.eps))
        return psnr, mse


class SSIM(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        channels: int = 3,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_range = data_range
        self.channels = channels
        self.k1 = k1
        self.k2 = k2
        self.eps = eps

        self.register_buffer("kernel", None)
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2

    def _init_kernel(self, device: torch.device):
        if self.kernel is None or self.kernel.device != device:
            self.kernel = create_gaussian_kernel(
                self.kernel_size, self.sigma, self.channels, device
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        self._init_kernel(pred.device)

        pred = (pred + 1.0) / 2.0 * self.data_range
        target = (target + 1.0) / 2.0 * self.data_range

        if mask is not None:
            mask = F.interpolate(mask, size=pred.shape[2:], mode="nearest")
            pred = pred * mask
            target = target * mask

        mu_x = F.conv2d(pred, self.kernel, padding=self.kernel_size//2, groups=self.channels)
        mu_y = F.conv2d(target, self.kernel, padding=self.kernel_size//2, groups=self.channels)

        sigma_x = F.conv2d(pred ** 2, self.kernel, padding=self.kernel_size//2, groups=self.channels) - mu_x ** 2
        sigma_y = F.conv2d(target ** 2, self.kernel, padding=self.kernel_size//2, groups=self.channels) - mu_y ** 2
        sigma_xy = F.conv2d(pred * target, self.kernel, padding=self.kernel_size//2, groups=self.channels) - mu_x * mu_y

        ssim_numerator = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)
        ssim_denominator = (mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2)
        ssim_map = ssim_numerator / (ssim_denominator + self.eps)

        if mask is not None:
            ssim_map = ssim_map * mask
            ssim = torch.sum(ssim_map, dim=[1, 2, 3]) / (torch.sum(mask, dim=[1, 2, 3]) + self.eps)
        else:
            ssim = torch.mean(ssim_map, dim=[1, 2, 3])

        return ssim


class PSNRSSIMMetrics(nn.Module):
    def __init__(
        self,
        data_range: float = 1.0,
        ssim_kernel_size: int = 11,
        ssim_sigma: float = 1.5,
        channels: int = 3
    ):
        super().__init__()
        self.psnr = PSNR(data_range=data_range)
        self.ssim = SSIM(
            kernel_size=ssim_kernel_size,
            sigma=ssim_sigma,
            data_range=data_range,
            channels=channels
        )

    @torch.no_grad()
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> Tuple[float, float]:
        psnr_per_sample, _ = self.psnr(pred, target, mask)
        ssim_per_sample = self.ssim(pred, target, mask)

        if reduction == "mean":
            return psnr_per_sample.mean().item(), ssim_per_sample.mean().item()
        elif reduction == "none":
            return psnr_per_sample, ssim_per_sample
        else:
            raise ValueError(f"Unsupported reduction mode: {reduction}")


# Runtime verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = PSNRSSIMMetrics().to(device)

    test_pred = torch.randn(4, 3, 256, 256).to(device)
    test_target = torch.randn(4, 3, 256, 256).to(device)
    test_mask = torch.zeros(4, 1, 256, 256).to(device)
    test_mask[:, :, 100:150, 100:150] = 1.0

    with torch.no_grad():
        psnr_mean, ssim_mean = metrics(test_pred, test_target, test_mask, reduction="mean")
        psnr_per, ssim_per = metrics(test_pred, test_target, test_mask, reduction="none")

    print(f"Mean PSNR: {psnr_mean:.4f} dB")
    print(f"Mean SSIM: {ssim_mean:.4f}")
    print(f"Per-sample PSNR shape: {psnr_per.shape}")
    print(f"Per-sample SSIM shape: {ssim_per.shape}")
    print("PSNR & SSIM metrics test completed.")
