import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .HMAM import HMAM


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        use_activation: bool = True
    ):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=not use_norm
        ))
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if use_activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        use_norm: bool = True,
        use_activation: bool = True
    ):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=not use_norm
        ))
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if use_activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv_block(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        base_channels: int = 64,
        num_heads: int = 6,
        d_head: int = 64,
        mem_dim: int = 256
    ):
        super().__init__()
        self.base_channels = base_channels

        # ===================== Encoder =====================
        self.enc1 = ConvBlock(in_channels, base_channels, stride=2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 4, dilation=2, padding=2)
        self.enc5 = ConvBlock(base_channels * 4, base_channels * 4, dilation=4, padding=4)
        self.enc6 = ConvBlock(base_channels * 4, base_channels * 4, dilation=8, padding=8)
        self.enc7 = ConvBlock(base_channels * 4, base_channels * 4, dilation=16, padding=16)

        # ===================== Decoder =====================
        self.dec1 = DeconvBlock(base_channels * 4, base_channels * 4)
        self.dec2 = DeconvBlock(base_channels * 8, base_channels * 4)
        self.dec3 = DeconvBlock(base_channels * 8, base_channels * 2)

        # HMAM for multi-scale feature fusion
        encoder_channels = [base_channels, base_channels * 2, base_channels * 4]
        self.hmam1 = HMAM(
            in_channels=base_channels * 2,
            multi_scale_channels=encoder_channels,
            num_heads=num_heads,
            d_head=d_head,
            mem_dim=mem_dim
        )
        self.dec4 = DeconvBlock(base_channels * 2, base_channels)

        self.hmam2 = HMAM(
            in_channels=base_channels,
            multi_scale_channels=encoder_channels,
            num_heads=num_heads,
            d_head=d_head,
            mem_dim=mem_dim
        )
        self.dec5 = DeconvBlock(base_channels, base_channels // 2)

        # Final output layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels // 2, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        init_img: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Input concatenation
        x = torch.cat([img, mask, init_img], dim=1)

        # Encoder forward with feature saving
        enc_feats = []
        x = self.enc1(x)
        enc_feats.append(x)
        x = self.enc2(x)
        enc_feats.append(x)
        x = self.enc3(x)
        enc_feats.append(x)

        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)

        # Decoder forward with skip connection
        x = self.dec1(x)
        x = torch.cat([x, enc_feats[-1]], dim=1)
        x = self.dec2(x)
        x = torch.cat([x, enc_feats[-2]], dim=1)
        x = self.dec3(x)

        # First HMAM with multi-scale encoder features
        x = self.hmam1(x, enc_feats)
        x = self.dec4(x)

        # Second HMAM with multi-scale encoder features
        x = self.hmam2(x, enc_feats)
        x = self.dec5(x)

        # Final output
        out = self.out_conv(x)

        return out, enc_feats


# Unit test for functionality verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)

    # Test input with 256x256 resolution
    batch_size = 2
    test_img = torch.randn(batch_size, 3, 256, 256).to(device)
    test_mask = torch.randn(batch_size, 1, 256, 256).to(device)
    test_init = torch.randn(batch_size, 3, 256, 256).to(device)

    with torch.no_grad():
        output, feats = generator(test_img, test_mask, test_init)

    print(f"Input image shape: {test_img.shape}")
    print(f"Generator output shape: {output.shape}")
    print(f"Encoder feature levels: {len(feats)}")
    for i, feat in enumerate(feats):
        print(f"Encoder layer {i+1} feature shape: {feat.shape}")

    # Gradient flow check
    generator.train()
    output_train, _ = generator(test_img, test_mask, test_init)
    loss = output_train.mean()
    loss.backward()

    grad_valid = True
    for name, param in generator.named_parameters():
        if param.requires_grad and (param.grad is None or torch.isnan(param.grad).any()):
            grad_valid = False
            break
    print(f"Gradient flow valid: {grad_valid}")
    print("Generator module test completed.")
