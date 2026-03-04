import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Tuple

__all__ = ["LPIPS"]


class LPIPS(nn.Module):
    def __init__(
        self,
        net_type: str = "vgg",
        pretrained: bool = True,
        eval_mode: bool = True,
        eps: float = 1e-8
    ):
        super().__init__()
        self.net_type = net_type
        self.eps = eps
        self.feature_layers = {
            "vgg": ["0", "5", "10", "19", "28"],
            "alex": ["0", "4", "8", "10", "12"]
        }[net_type]

        # Backbone network initialization
        if net_type == "vgg":
            self.backbone = models.vgg19(pretrained=pretrained).features
        elif net_type == "alex":
            self.backbone = models.alexnet(pretrained=pretrained).features
        else:
            raise ValueError(f"Unsupported network type: {net_type}")

        # Channel weights for each feature layer
        self.channel_weights = nn.ParameterList()
        self._init_channel_weights()

        # Input normalization parameters
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

    def _init_channel_weights(self):
        channel_dims = {
            "vgg": [64, 128, 256, 512, 512],
            "alex": [64, 192, 384, 256, 256]
        }[self.net_type]
        for dim in channel_dims:
            self.channel_weights.append(nn.Parameter(torch.ones(1, dim, 1, 1)))

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for name, module in self.backbone.named_children():
            x = module(x)
            if name in self.feature_layers:
                features.append(x)
        return features

    def _normalize_features(self, feat: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(feat ** 2, dim=1, keepdim=True) + self.eps)
        return feat / norm

    @torch.no_grad()
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> torch.Tensor:
        B, C, H, W = pred.shape

        # Input normalization
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Feature extraction
        pred_feats = self._extract_features(pred)
        target_feats = self._extract_features(target)

        # LPIPS calculation
        lpips_per_sample = torch.zeros(B, device=pred.device)
        for i, (pred_feat, target_feat, weight) in enumerate(zip(pred_feats, target_feats, self.channel_weights)):
            pred_norm = self._normalize_features(pred_feat)
            target_norm = self._normalize_features(target_feat)
            sq_diff = (pred_norm - target_norm) ** 2

            if mask is not None:
                mask_scaled = F.interpolate(mask, size=sq_diff.shape[2:], mode="nearest")
                sq_diff = sq_diff * mask_scaled
                spatial_sum = torch.sum(sq_diff * weight, dim=[1, 2, 3])
                spatial_sum = spatial_sum / (torch.sum(mask_scaled, dim=[1, 2, 3]) + self.eps)
            else:
                spatial_sum = torch.mean(sq_diff * weight, dim=[1, 2, 3])

            lpips_per_sample += spatial_sum

        if reduction == "mean":
            return lpips_per_sample.mean().item()
        elif reduction == "none":
            return lpips_per_sample
        else:
            raise ValueError(f"Unsupported reduction mode: {reduction}")


# Runtime verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_metric = LPIPS(net_type="vgg").to(device)

    test_pred = torch.randn(4, 3, 256, 256).to(device)
    test_target = torch.randn(4, 3, 256, 256).to(device)
    test_mask = torch.zeros(4, 1, 256, 256).to(device)
    test_mask[:, :, 100:150, 100:150] = 1.0

    with torch.no_grad():
        lpips_mean = lpips_metric(test_pred, test_target, test_mask, reduction="mean")
        lpips_per = lpips_metric(test_pred, test_target, test_mask, reduction="none")

    print(f"Mean LPIPS: {lpips_mean:.6f}")
    print(f"Per-sample LPIPS shape: {lpips_per.shape}")
    print("LPIPS metric test completed.")
