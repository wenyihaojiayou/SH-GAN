import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from .perceptual import PerceptualLoss
from .adaptive_weight import HierarchicalWeightAdaptiveLoss

__all__ = ["GeneratorLoss"]


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        lambda_adv: float = 0.1,
        lambda_rec: float = 1.0,
        lambda_style: float = 120.0,
        lambda_color: float = 50.0
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_rec = lambda_rec
        self.lambda_style = lambda_style
        self.lambda_color = lambda_color

        self.perceptual_loss = PerceptualLoss()
        self.hw_adaptive = HierarchicalWeightAdaptiveLoss()

    def compute_adv_loss(self, disc_outputs: List[torch.Tensor]) -> torch.Tensor:
        adv_loss = 0.0
        for out in disc_outputs:
            adv_loss += -torch.mean(out)
        return adv_loss / len(disc_outputs)

    def forward(
        self,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor,
        mask: torch.Tensor,
        disc_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Perceptual sub-losses
        rec_loss, style_loss, color_loss = self.perceptual_loss(pred_img, gt_img, mask)
        rec_loss = rec_loss * self.lambda_rec
        style_loss = style_loss * self.lambda_style
        color_loss = color_loss * self.lambda_color

        # Hierarchical weighted adaptive fusion
        recon_loss, w_r, w_s, w_k = self.hw_adaptive(rec_loss, style_loss, color_loss)

        # Adversarial loss
        adv_loss = self.compute_adv_loss(disc_outputs)
        adv_loss = adv_loss * self.lambda_adv

        # Total generator loss
        total_loss = recon_loss + adv_loss

        # Loss log dict
        loss_log = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "adv_loss": adv_loss.item(),
            "rec_loss": rec_loss.item(),
            "style_loss": style_loss.item(),
            "color_loss": color_loss.item(),
            "w_rec": w_r.item(),
            "w_style": w_s.item(),
            "w_color": w_k.item()
        }

        return total_loss, loss_log


# Runtime verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_loss = GeneratorLoss().to(device)

    test_pred = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)
    test_gt = torch.randn(2, 3, 256, 256, device=device)
    test_mask = torch.zeros(2, 1, 256, 256, device=device)
    test_mask[:, :, 100:150, 100:150] = 1.0
    test_disc_outs = [
        torch.randn(2, 1, 30, 30, device=device),
        torch.randn(2, 1, 30, 30, device=device),
        torch.randn(2, 1, 15, 15, device=device),
        torch.randn(2, 1, 7, 7, device=device)
    ]

    total_loss, loss_log = gen_loss(test_pred, test_gt, test_mask, test_disc_outs)
    total_loss.backward()

    print(f"Total generator loss: {loss_log['total_loss']:.6f}")
    print(f"Reconstruction loss: {loss_log['recon_loss']:.6f}")
    print(f"Adversarial loss: {loss_log['adv_loss']:.6f}")
    print(f"Gradient valid: {test_pred.grad is not None and not torch.isnan(test_pred.grad).any()}")
    print("Generator loss module test completed.")
