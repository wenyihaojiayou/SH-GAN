import torch
import torch.nn as nn
from typing import Tuple

__all__ = ["HierarchicalWeightAdaptiveLoss"]


class HierarchicalWeightAdaptiveLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        rec_loss: torch.Tensor,
        style_loss: torch.Tensor,
        color_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = rec_loss + style_loss + color_loss + self.eps

        w_rec = rec_loss / total
        w_style = style_loss / total
        w_color = color_loss / total

        weighted_loss = w_rec * rec_loss + w_style * style_loss + w_color * color_loss

        return weighted_loss, w_rec, w_style, w_color


# Runtime verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw_adaptive = HierarchicalWeightAdaptiveLoss().to(device)

    test_rec = torch.tensor(10.0, device=device)
    test_style = torch.tensor(5.0, device=device)
    test_color = torch.tensor(2.0, device=device)

    weighted_loss, w_r, w_s, w_k = hw_adaptive(test_rec, test_style, test_color)

    print(f"Weighted total loss: {weighted_loss.item():.6f}")
    print(f"Content weight: {w_r.item():.4f}, Style weight: {w_s.item():.4f}, Color weight: {w_k.item():.4f}")
    print(f"Weight sum: {(w_r + w_s + w_k).item():.4f}")
    print("Adaptive weight module test completed.")
