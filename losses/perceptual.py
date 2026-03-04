import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple

__all__ = ["VGGFeatureExtractor", "LabConverter", "PerceptualLoss"]


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_layers: List[int] = None,
        requires_grad: bool = False
    ):
        super().__init__()
        if feature_layers is None:
            feature_layers = [0, 5, 10, 19, 28]
        self.feature_layers = feature_layers

        vgg = models.vgg19(pretrained=True).features
        self.blocks = nn.ModuleList()
        prev_idx = 0
        for layer_idx in sorted(feature_layers):
            block = nn.Sequential()
            for i in range(prev_idx, layer_idx + 1):
                block.add_module(str(i), vgg[i])
            self.blocks.append(block)
            prev_idx = layer_idx + 1

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class LabConverter(nn.Module):
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.register_buffer(
            "xyz_rgb",
            torch.tensor([
                [0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]
            ], device=device)
        )
        self.register_buffer(
            "rgb_xyz",
            torch.tensor([
                [3.240479, -1.537150, -0.498535],
                [-0.969256, 1.875992, 0.041556],
                [0.055648, -0.204043, 1.057311]
            ], device=device)
        )
        self.register_buffer("xyz_white", torch.tensor([0.95047, 1.0, 1.08883], device=device))
        self.eps = 1e-8

    def _f(self, t: torch.Tensor) -> torch.Tensor:
        return torch.where(
            t > 0.008856,
            torch.pow(t, 1/3),
            t * 7.787 + 16/116
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        B, C, H, W = x.shape

        rgb = x.permute(0, 2, 3, 1).reshape(-1, 3)
        mask = rgb > 0.04045
        rgb = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

        xyz = torch.matmul(rgb, self.rgb_xyz.T)
        xyz = xyz / self.xyz_white.view(1, 3)

        fx = self._f(xyz[:, 0])
        fy = self._f(xyz[:, 1])
        fz = self._f(xyz[:, 2])

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        lab = torch.stack([L, a, b], dim=-1)
        lab = lab.view(B, H, W, 3).permute(0, 3, 1, 2)
        return lab


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_extractor = VGGFeatureExtractor()
        self.lab_converter = LabConverter()
        self.eps = 1e-8

    def compute_content_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        mask = F.interpolate(mask, size=pred.shape[2:], mode="nearest")
        diff = mask * (pred - target)
        loss = torch.sum(diff ** 2) / (torch.sum(mask) + self.eps)
        return loss

    def compute_style_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred_feats = self.vgg_extractor(pred)
        target_feats = self.vgg_extractor(target)

        style_loss = 0.0
        for pred_feat, target_feat in zip(pred_feats, target_feats):
            B, C, H, W = pred_feat.shape
            pred_flat = pred_feat.view(B, C, H * W)
            target_flat = target_feat.view(B, C, H * W)

            pred_gram = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / (C * H * W)
            target_gram = torch.bmm(target_flat, target_flat.transpose(1, 2)) / (C * H * W)

            style_loss += torch.mean((pred_gram - target_gram) ** 2)

        return style_loss / len(pred_feats)

    def compute_color_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred_lab = self.lab_converter(pred)
        target_lab = self.lab_converter(target)
        loss = torch.mean(torch.abs(pred_lab - target_lab))
        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec_loss = self.compute_content_loss(pred, target, mask)
        style_loss = self.compute_style_loss(pred, target)
        color_loss = self.compute_color_loss(pred, target)
        return rec_loss, style_loss, color_loss


# Runtime verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual_loss = PerceptualLoss().to(device)

    test_pred = torch.randn(2, 3, 256, 256).to(device)
    test_target = torch.randn(2, 3, 256, 256).to(device)
    test_mask = torch.zeros(2, 1, 256, 256).to(device)
    test_mask[:, :, 100:150, 100:150] = 1.0

    with torch.no_grad():
        rec, style, color = perceptual_loss(test_pred, test_target, test_mask)

    print(f"Content loss: {rec.item():.6f}")
    print(f"Style loss: {style.item():.6f}")
    print(f"Color loss: {color.item():.6f}")
    print("Perceptual loss module test completed.")
