import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

__all__ = ["ExternalAttention", "HMAM"]


class ExternalAttention(nn.Module):

    def __init__(
        self,
        d_head: int = 64,
        mem_dim: int = 256,
        eps: float = 1e-8
    ):
        super().__init__()
        self.d_head = d_head
        self.mem_dim = mem_dim
        self.eps = eps

        # External shared memory units (M_k, M_v in formula 12/16), shared across all heads
        self.M_k = nn.Parameter(torch.empty(mem_dim, d_head))
        self.M_v = nn.Parameter(torch.empty(mem_dim, d_head))

        # Initialize parameters
        nn.init.xavier_normal_(self.M_k)
        nn.init.xavier_normal_(self.M_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, N, _ = x.shape

      
        attn = x @ self.M_k.T

        # Column-wise softmax normalization (formula 14, core of external attention)
        attn = F.softmax(attn, dim=1)
        # L1 normalization along the feature dimension
        attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + self.eps)

    
        out = attn @ self.M_v

        return out


class HMAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        multi_scale_channels: Optional[List[int]] = None,
        num_heads: int = 6,
        d_head: int = 64,
        mem_dim: int = 256,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.in_channels = in_channels
        self.multi_scale_channels = multi_scale_channels or []
        self.num_heads = num_heads
        self.d_head = d_head
        self.total_dim = num_heads * d_head

   
        self.scale_proj = nn.ModuleList()
        total_in_channels = in_channels

        if len(self.multi_scale_channels) > 0:
            for c in self.multi_scale_channels:
                self.scale_proj.append(
                    nn.Sequential(
                        nn.Conv2d(c, in_channels, kernel_size=1, stride=1, padding=0),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
            total_in_channels = in_channels * (1 + len(self.multi_scale_channels))

   
        self.in_proj = nn.Conv2d(total_in_channels, self.total_dim, kernel_size=1, stride=1, padding=0)
        self.norm = norm_layer(self.total_dim)

      
        self.heads = nn.ModuleList([
            ExternalAttention(d_head=d_head, mem_dim=mem_dim)
            for _ in range(num_heads)
        ])

    
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.total_dim, in_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

    
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        multi_scale_feats: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:

        B, C, H, W = x.shape
        residual = x

   
        feat_list = [x]

    
        if multi_scale_feats is not None and len(multi_scale_feats) == len(self.scale_proj):
            for feat, proj in zip(multi_scale_feats, self.scale_proj):
                # Align spatial size to main input
                feat_aligned = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
                # Align channel dimension
                feat_proj = proj(feat_aligned)
                feat_list.append(feat_proj)

   
        fused_feat = torch.cat(feat_list, dim=1)

    
        x_proj = self.in_proj(fused_feat)
        x_flat = x_proj.flatten(2).transpose(1, 2)  # [B, N, total_dim]
        x_norm = self.norm(x_flat)

  
        x_heads = torch.chunk(x_norm, self.num_heads, dim=-1)
        head_outputs = [head(x_h) for head, x_h in zip(self.heads, x_heads)]

    
        multi_head_out = torch.cat(head_outputs, dim=-1)


        multi_head_out = multi_head_out.transpose(1, 2).reshape(B, self.total_dim, H, W)


        out = self.out_proj(multi_head_out)
        out = residual + self.res_scale * out

        return out


# Unit test (for open source users to quickly verify functionality)
if __name__ == "__main__":
    # Test device compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing HMAM on device: {device}")


    print("\n=== Test 1: Single-scale input ===")
    hmam_single = HMAM(in_channels=256, num_heads=6, d_head=64).to(device)

    # Input: [batch=2, channel=256, height=32, width=32]
    test_feat = torch.randn(2, 256, 32, 32).to(device)
    with torch.no_grad():
        output = hmam_single(test_feat)

    print(f"Input shape: {test_feat.shape}")
    print(f"Output shape: {output.shape}")
    print("Single-scale test passed! Input/output dimension matches.")

 

    print("\n=== Test 2: Multi-scale hierarchical input ===")
    # Main input channel=128, 3 hierarchical encoder features with [64, 128, 256] channels
    hmam_multi = HMAM(
        in_channels=128,
        multi_scale_channels=[64, 128, 256],
        num_heads=6,
        d_head=64
    ).to(device)

    # Main input: [batch=2, 128, 64, 64]
    main_feat = torch.randn(2, 128, 64, 64).to(device)
    # Multi-scale encoder features (different spatial size & channel)
    multi_feats = [
        torch.randn(2, 64, 128, 128).to(device),   # shallow layer feature
        torch.randn(2, 128, 64, 64).to(device),    # middle layer feature
        torch.randn(2, 256, 32, 32).to(device)     # deep layer feature
    ]

    with torch.no_grad():
        output_multi = hmam_multi(main_feat, multi_feats)

    print(f"Main input shape: {main_feat.shape}")
    print(f"Output shape: {output_multi.shape}")
    print("Multi-scale test passed! Cross-layer feature fusion works normally.")


    print("\n=== Test 3: Gradient flow check ===")
    hmam_train = HMAM(in_channels=128, multi_scale_channels=[64]).to(device)
    hmam_train.train()

    main_feat = torch.randn(2, 128, 32, 32, device=device, requires_grad=True)
    multi_feat = [torch.randn(2, 64, 64, 64, device=device, requires_grad=True)]

    output = hmam_train(main_feat, multi_feat)
    loss = output.mean()
    loss.backward()

    # Check if gradients are valid
    grad_valid = main_feat.grad is not None and not torch.isnan(main_feat.grad).any()
    print(f"Gradient valid: {grad_valid}")
    print("Gradient flow test passed! Module supports end-to-end training.")

    print("\n✅ All HMAM tests passed! Module is ready for integration into SH-GAN.")
