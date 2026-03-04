import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Tuple, Dict

__all__ = ["DiscriminatorLoss"]


class DiscriminatorLoss(nn.Module):
    def __init__(
        self,
        lambda_gp: float = 10.0,
        lambda_mix: float = 0.1
    ):
        super().__init__()
        self.lambda_gp = lambda_gp
        self.lambda_mix = lambda_mix
        self.eps = 1e-8

    def compute_gradient_penalty(
        self,
        disc: nn.Module,
        real_img: torch.Tensor,
        fake_img: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = real_img.shape
        alpha = torch.rand(B, 1, 1, 1, device=real_img.device)
        interpolates = (alpha * real_img + (1 - alpha) * fake_img).requires_grad_(True)

        disc_interp, _ = disc(interpolates, mask)
        disc_interp = sum([torch.mean(out) for out in disc_interp])

        gradients = autograd.grad(
            outputs=disc_interp,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(B, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + self.eps)
        gp = torch.mean((gradient_norm - 1) ** 2)
        return gp

    def compute_mix_loss(
        self,
        real_feats: List[torch.Tensor],
        fake_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        mix_loss = 0.0
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            B, C, H, W = real_feat.shape
            real_flat = real_feat.view(B, -1)
            fake_flat = fake_feat.view(B, -1)

            alpha = torch.rand(B, 1, device=real_feat.device)
            mix_feat = alpha * real_flat + (1 - alpha) * fake_flat

            dist_real = torch.cdist(mix_feat, real_flat, p=2)
            dist_fake = torch.cdist(mix_feat, fake_flat, p=2)

            pos_loss = torch.mean(torch.min(dist_real, dim=1)[0])
            neg_loss = torch.mean(torch.min(dist_fake, dim=1)[0])
            mix_loss += torch.relu(pos_loss - neg_loss + 0.1)

        return mix_loss / len(real_feats)

    def forward(
        self,
        disc: nn.Module,
        real_img: torch.Tensor,
        fake_img: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Discriminator forward
        real_outs, real_feats = disc(real_img, mask)
        fake_outs, fake_feats = disc(fake_img.detach(), mask)

        # WGAN-GP adversarial loss
        adv_loss = 0.0
        for real_out, fake_out in zip(real_outs, fake_outs):
            adv_loss += torch.mean(fake_out) - torch.mean(real_out)
        adv_loss = adv_loss / len(real_outs)

        # Gradient penalty
        gp = self.compute_gradient_penalty(disc, real_img, fake_img, mask)
        gp_loss = self.lambda_gp * gp

        # Adaptive mixing loss
        mix_loss = self.compute_mix_loss(real_feats, fake_feats)
        mix_loss = self.lambda_mix * mix_loss

        # Total discriminator loss
        total_loss = adv_loss + gp_loss + mix_loss

        # Loss log dict
        loss_log = {
            "total_loss": total_loss.item(),
            "adv_loss": adv_loss.item(),
            "gp_loss": gp_loss.item(),
            "mix_loss": mix_loss.item()
        }

        return total_loss, loss_log


# Runtime verification
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.discriminator import MultiBranchDiscriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc = MultiBranchDiscriminator().to(device)
    disc_loss = DiscriminatorLoss().to(device)

    test_real = torch.randn(2, 3, 256, 256, device=device)
    test_fake = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)
    test_mask = torch.zeros(2, 1, 256, 256, device=device)
    test_mask[:, :, 100:150, 100:150] = 1.0

    total_loss, loss_log = disc_loss(disc, test_real, test_fake, test_mask)
    total_loss.backward()

    print(f"Total discriminator loss: {loss_log['total_loss']:.6f}")
    print(f"Adversarial loss: {loss_log['adv_loss']:.6f}")
    print(f"Gradient penalty loss: {loss_log['gp_loss']:.6f}")
    print(f"Mixing loss: {loss_log['mix_loss']:.6f}")

    grad_valid = True
    for name, param in disc.named_parameters():
        if param.requires_grad and (param.grad is None or torch.isnan(param.grad).any()):
            grad_valid = False
            break
    print(f"Discriminator gradient valid: {grad_valid}")
    print("Discriminator loss module test completed.")
