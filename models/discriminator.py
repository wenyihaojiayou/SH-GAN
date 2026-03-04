import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from .hmam import HMAM


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
        use_activation: bool = True
    ):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=not use_norm
        ))
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if use_activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class BaseDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        use_attention: bool = False,
        num_heads: int = 6,
        d_head: int = 64,
        mem_dim: int = 256
    ):
        super().__init__()
        self.use_attention = use_attention

        # Backbone downsampling layers
        self.enc1 = ConvBlock(in_channels, base_channels, use_norm=False)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.enc5 = ConvBlock(base_channels * 8, base_channels * 8, stride=1)

        # HMAM for global context modeling
        if self.use_attention:
            self.hmam = HMAM(
                in_channels=base_channels * 8,
                multi_scale_channels=None,
                num_heads=num_heads,
                d_head=d_head,
                mem_dim=mem_dim
            )

        # Final output projection
        self.out_conv = nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward through backbone
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        feat = self.enc5(x)

        # Attention fusion if enabled
        if self.use_attention:
            feat = self.hmam(feat)

        # Final discriminator output
        out = self.out_conv(feat)

        return out, feat


class MultiBranchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_heads: int = 6,
        d_head: int = 64,
        mem_dim: int = 256,
        crop_size: int = 256
    ):
        super().__init__()
        self.crop_size = crop_size

        # Global structural discriminator (D_g)
        self.global_disc = BaseDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            use_attention=True,
            num_heads=num_heads,
            d_head=d_head,
            mem_dim=mem_dim
        )

        # Local restoration discriminator (D_l)
        self.local_disc = BaseDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            use_attention=True,
            num_heads=num_heads,
            d_head=d_head,
            mem_dim=mem_dim
        )

        # Multi-scale semantic discriminator 1 (D_a1, 1/2 resolution)
        self.aug_disc_1 = BaseDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            use_attention=False
        )

        # Multi-scale semantic discriminator 2 (D_a2, 1/4 resolution)
        self.aug_disc_2 = BaseDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            use_attention=False
        )

    def crop_local_region(
        self,
        img: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = img.shape
        crop_size = self.crop_size
        local_crops = []

        for b in range(B):
            # Get mask coordinates for current sample
            mask_b = mask[b, 0]
            damaged_coords = torch.nonzero(mask_b > 0.5, as_tuple=True)

            # Handle non-damaged case
            if len(damaged_coords[0]) == 0:
                crop = img[b, :, :crop_size, :crop_size]
                local_crops.append(crop)
                continue

            # Get bounding box of damaged region
            y_min, y_max = damaged_coords[0].min(), damaged_coords[0].max()
            x_min, x_max = damaged_coords[1].min(), damaged_coords[1].max()

            # Calculate crop center and expand context
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            half_crop = crop_size // 2

            # Adjust crop range to avoid out of bounds
            start_y = torch.clip(center_y - half_crop, 0, H - crop_size)
            start_x = torch.clip(center_x - half_crop, 0, W - crop_size)
            end_y = start_y + crop_size
            end_x = start_x + crop_size

            # Crop and collect
            crop = img[b, :, start_y:end_y, start_x:end_x]
            local_crops.append(crop)

        # Stack into batch tensor
        local_batch = torch.stack(local_crops, dim=0)
        return local_batch

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Crop local region for local discriminator
        local_img = self.crop_local_region(img, mask)

        # Global discriminator forward
        global_out, global_feat = self.global_disc(img)

        # Local discriminator forward
        local_out, local_feat = self.local_disc(local_img)

        # Multi-scale discriminator forward (1/2 resolution)
        img_half = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
        aug1_out, aug1_feat = self.aug_disc_1(img_half)

        # Multi-scale discriminator forward (1/4 resolution)
        img_quarter = F.interpolate(img, scale_factor=0.25, mode='bilinear', align_corners=False)
        aug2_out, aug2_feat = self.aug_disc_2(img_quarter)

        # Collect outputs and intermediate features
        outputs = [global_out, local_out, aug1_out, aug2_out]
        features = [global_feat, local_feat, aug1_feat, aug2_feat]

        return outputs, features


# Runtime verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = MultiBranchDiscriminator().to(device)

    # Test input with 256x256 resolution
    batch_size = 2
    test_img = torch.randn(batch_size, 3, 256, 256).to(device)
    test_mask = torch.zeros(batch_size, 1, 256, 256).to(device)
    test_mask[:, :, 100:150, 100:150] = 1.0

    with torch.no_grad():
        outputs, features = discriminator(test_img, test_mask)

    # Print dimension check
    branch_names = ["Global", "Local", "Aug1(1/2)", "Aug2(1/4)"]
    for name, out, feat in zip(branch_names, outputs, features):
        print(f"{name} Discriminator output shape: {out.shape}")
        print(f"{name} Discriminator feature shape: {feat.shape}")

    # Gradient flow check
    discriminator.train()
    outputs_train, _ = discriminator(test_img, test_mask)
    loss = sum([out.mean() for out in outputs_train])
    loss.backward()

    grad_valid = True
    for name, param in discriminator.named_parameters():
        if param.requires_grad and (param.grad is None or torch.isnan(param.grad).any()):
            grad_valid = False
            break
    print(f"\nGradient flow valid: {grad_valid}")
    print("Multi-branch discriminator module test completed.")
