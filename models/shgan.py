import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .shtm import SHTM
from .generator import Generator
from .discriminator import MultiBranchDiscriminator


class SHGAN(nn.Module):
    def __init__(
        self,
        # SHTM parameters
        block_size: int = 8,
        search_window: int = 21,
        topk_blocks: int = 8,
        topk_rows: int = 4,
        tau_valid_ratio: float = 0.4,
        freeze_shtm: bool = False,
        # Generator parameters
        gen_in_channels: int = 7,
        gen_base_channels: int = 64,
        num_heads: int = 6,
        d_head: int = 64,
        mem_dim: int = 256,
        # Discriminator parameters
        disc_in_channels: int = 3,
        disc_base_channels: int = 64,
        crop_size: int = 256,
    ):
        super().__init__()

        # Self-similarity Haar Transform Module
        self.shtm = SHTM(
            block_size=block_size,
            search_window=search_window,
            topk_blocks=topk_blocks,
            topk_rows=topk_rows,
            tau_valid_ratio=tau_valid_ratio
        )
        if freeze_shtm:
            for param in self.shtm.parameters():
                param.requires_grad = False

        # Multi-scale feature feedback generator
        self.generator = Generator(
            in_channels=gen_in_channels,
            base_channels=gen_base_channels,
            num_heads=num_heads,
            d_head=d_head,
            mem_dim=mem_dim
        )

        # Multi-branch discriminator group
        self.discriminator = MultiBranchDiscriminator(
            in_channels=disc_in_channels,
            base_channels=disc_base_channels,
            num_heads=num_heads,
            d_head=d_head,
            mem_dim=mem_dim,
            crop_size=crop_size
        )

    def shtm_initialize(
        self,
        img: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.set_grad_enabled(self.shtm.training and not self._is_shtm_frozen()):
            init_img = self.shtm(img, mask)
        return init_img

    def _is_shtm_frozen(self) -> bool:
        for param in self.shtm.parameters():
            if param.requires_grad:
                return False
        return True

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        return_feats: bool = False,
        init_img: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor]:
        # Structural initialization
        if init_img is None:
            init_img = self.shtm_initialize(img, mask)

        # Generator forward
        pred_img, enc_feats = self.generator(img, mask, init_img)

        if return_feats:
            return pred_img, enc_feats, init_img
        else:
            return pred_img, None, init_img

    def discriminate(
        self,
        img: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        disc_outputs, disc_feats = self.discriminator(img, mask)
        return disc_outputs, disc_feats

    def get_generator_params(self):
        return list(self.shtm.parameters()) + list(self.generator.parameters())

    def get_discriminator_params(self):
        return list(self.discriminator.parameters())


# Runtime verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SH-GAN on device: {device}")

    # Initialize full model
    model = SHGAN(freeze_shtm=False).to(device)

    # Test input
    batch_size = 2
    test_img = torch.randn(batch_size, 3, 256, 256).to(device)
    test_mask = torch.zeros(batch_size, 1, 256, 256).to(device)
    test_mask[:, :, 80:180, 80:180] = 1.0

    # Inference forward test
    model.eval()
    with torch.no_grad():
        pred_img, _, init_img = model(test_img, test_mask, return_feats=False)
        print(f"Input damaged image shape: {test_img.shape}")
        print(f"SHTM initialized image shape: {init_img.shape}")
        print(f"Generator restored image shape: {pred_img.shape}")

    # Discriminator forward test
    disc_outputs, disc_feats = model.discriminate(pred_img, test_mask)
    branch_names = ["Global", "Local", "Aug1", "Aug2"]
    for name, out in zip(branch_names, disc_outputs):
        print(f"{name} discriminator output shape: {out.shape}")

    # Training gradient flow test
    model.train()
    pred_train, enc_feats, _ = model(test_img, test_mask, return_feats=True)
    gen_loss = pred_train.mean()
    gen_loss.backward()

    gen_grad_valid = True
    for name, param in model.generator.named_parameters():
        if param.requires_grad and (param.grad is None or torch.isnan(param.grad).any()):
            gen_grad_valid = False
            break

    shtm_grad_valid = True
    for name, param in model.shtm.named_parameters():
        if param.requires_grad and (param.grad is None or torch.isnan(param.grad).any()):
            shtm_grad_valid = False
            break

    print(f"\nGenerator gradient valid: {gen_grad_valid}")
    print(f"SHTM gradient valid: {shtm_grad_valid}")
    print("SH-GAN full model test completed.")
