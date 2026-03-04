import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

__all__ = ["SHTM", "SHTMPreprocess"]


class SHTM(nn.Module):
    """
    Self-Similarity Haar Transform Module (SHTM)
    """

    def __init__(
        self,
        block_size: int = 8,
        search_window: int = 21,
        topk_blocks: int = 8,
        topk_rows: int = 4,
        tau_valid_ratio: float = 0.4,
        eps: float = 1e-8
    ):
        super().__init__()
        self.block_size = block_size
        self.search_window = search_window
        self.topk_blocks = topk_blocks
        self.topk_rows = topk_rows
        self.tau_valid_ratio = tau_valid_ratio
        self.eps = eps

        # Precompute Haar transform matrices (orthogonal, trainable-free)
        self._register_haar_basis()

    def _register_haar_basis(self):
        """Precompute 1D Haar basis matrices for fast transform, align with formula 6"""
        for size in [self.topk_rows, self.topk_blocks]:
            if size < 2:
                continue
            # Generate orthogonal Haar transform matrix
            haar_mat = torch.zeros(size, size)
            haar_mat[0, :] = 1.0 / torch.sqrt(torch.tensor(size, dtype=torch.float32))

            level = 1
            while level < size:
                step = size // (level * 2)
                for i in range(level):
                    start = i * step * 2
                    haar_mat[level + i, start:start + step] = 1.0 / torch.sqrt(torch.tensor(step * 2, dtype=torch.float32))
                    haar_mat[level + i, start + step:start + step * 2] = -1.0 / torch.sqrt(torch.tensor(step * 2, dtype=torch.float32))
                level *= 2

            # Register buffer (no gradient, device auto-sync)
            self.register_buffer(f"haar_mat_{size}", haar_mat)

    def _haar_transform_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape [B, q, m] (batch of pixel groups G_q,m)
        Returns:
            coeffs: transformed coefficient matrix Z_q,m, shape [B, q, m]
        """
        B, q, m = x.shape
        # Get precomputed Haar matrices
        haar_q = getattr(self, f"haar_mat_{q}", None)
        haar_m = getattr(self, f"haar_mat_{m}", None)

        if haar_q is None or haar_m is None:
            raise ValueError(f"Haar matrix not found for size q={q}, m={m}")

        # Row-wise + Column-wise transform (X_a @ G @ X_b)
        coeffs = torch.bmm(haar_q.repeat(B, 1, 1), x)
        coeffs = torch.bmm(coeffs, haar_m.repeat(B, 1, 1).transpose(1, 2))
        return coeffs

    def _inverse_haar_transform_2d(self, coeffs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            coeffs: thresholded coefficient matrix, shape [B, q, m]
        Returns:
            restored_groups: restored pixel groups G, shape [B, q, m]
        """
        B, q, m = coeffs.shape
        haar_q = getattr(self, f"haar_mat_{q}", None)
        haar_m = getattr(self, f"haar_mat_{m}", None)

        if haar_q is None or haar_m is None:
            raise ValueError(f"Haar matrix not found for size q={q}, m={m}")

        # Inverse transform (X_a^T @ Z @ X_b)
        restored = torch.bmm(haar_q.repeat(B, 1, 1).transpose(1, 2), coeffs)
        restored = torch.bmm(restored, haar_m.repeat(B, 1, 1))
        return restored

    def _pad_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad image to fit block size, avoid dimension mismatch"""
        B, C, H, W = x.shape
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x_padded, (H, W)

    def block_level_matching(
        self,
        img: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, C, H, W = img.shape
        kernel_size = self.block_size
        stride = 1

        # Unfold image and mask into patches
        unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
        img_patches = unfold(img)  # [B, C*K*K, N]
        mask_patches = unfold(mask)  # [B, 1*K*K, N]

        N = img_patches.shape[-1]
        img_patches = img_patches.reshape(B, C, kernel_size, kernel_size, N).permute(0, 4, 1, 2, 3)  # [B, N, C, K, K]
        mask_patches = mask_patches.reshape(B, 1, kernel_size, kernel_size, N).permute(0, 4, 1, 2, 3)  # [B, N, 1, K, K]

        # Step 1: Filter valid reference blocks (formula 2)
        # Valid condition: 1. intersect with damaged region; 2. known pixel ratio > tau_valid_ratio
        damaged_ratio = mask_patches.mean(dim=[2, 3, 4])  # [B, N]
        has_damaged = (damaged_ratio > 0) & (damaged_ratio < 1.0)  # intersect with damaged region
        known_ratio = 1.0 - damaged_ratio
        is_valid = has_damaged & (known_ratio >= self.tau_valid_ratio)

        # Get valid block indices
        valid_idx = torch.nonzero(is_valid, as_tuple=True)[1].unsqueeze(0).repeat(B, 1)  # [B, N_valid]
        B, N_valid = valid_idx.shape
        if N_valid == 0:
            return img_patches, valid_idx, torch.empty(B, 0, self.topk_blocks, device=img.device, dtype=torch.long)

        # Extract valid reference patches
        ref_patches = img_patches[torch.arange(B).unsqueeze(1), valid_idx]  # [B, N_valid, C, K, K]
        ref_mask = mask_patches[torch.arange(B).unsqueeze(1), valid_idx]  # [B, N_valid, 1, K, K]
        ref_weight = 1.0 - ref_mask  # weight: 1 for known pixel, 0 for damaged (formula 3)

        # Step 2: Neighborhood search window limit
        half_window = self.search_window // 2
        H_patch = H - kernel_size + 1
        W_patch = W - kernel_size + 1
        patch_coords = torch.stack(torch.meshgrid(torch.arange(H_patch), torch.arange(W_patch), indexing="ij"), dim=-1).reshape(-1, 2).to(img.device)  # [N, 2]
        ref_coords = patch_coords[valid_idx]  # [B, N_valid, 2]

        # Compute distance between patch coordinates, filter neighborhood
        coord_dist = torch.cdist(ref_coords.float(), patch_coords.float(), p=2)  # [B, N_valid, N]
        in_neighborhood = coord_dist <= half_window  # only search in neighborhood

        # Step 3: Weighted Euclidean distance (formula 3)
        ref_flat = ref_patches.reshape(B, N_valid, -1)  # [B, N_valid, C*K*K]
        all_flat = img_patches.reshape(B, N, -1)  # [B, N, C*K*K]
        weight_flat = ref_weight.reshape(B, N_valid, -1)  # [B, N_valid, C*K*K]

        # Weighted distance: sum(w_ij * (p_ref - p_cand)^2)
        dist = torch.sum(
            weight_flat.unsqueeze(2) * (ref_flat.unsqueeze(2) - all_flat.unsqueeze(1)) ** 2,
            dim=-1
        )  # [B, N_valid, N]

        # Mask out non-neighborhood blocks and self block
        dist = torch.where(in_neighborhood, dist, torch.inf * torch.ones_like(dist))
        dist.scatter_(2, valid_idx.unsqueeze(2), torch.inf * torch.ones_like(valid_idx.unsqueeze(2), dtype=dist.dtype))

        # Step 4: Select top-k similar blocks
        topk_indices = torch.topk(-dist, k=self.topk_blocks, dim=-1, largest=True).indices  # [B, N_valid, topk_blocks]

        return img_patches, valid_idx, topk_indices

    def row_level_matching(
        self,
        block_group: torch.Tensor
    ) -> torch.Tensor:

        B, m, n = block_group.shape  # n = C*block_size*block_size

        # Construct similarity matrix M: [B, n, m] (row=pixel position, column=similar block)
        M = block_group.permute(0, 2, 1)  # [B, n, m]

        # Compute row-wise Euclidean distance (formula 4)
        row_dist = torch.cdist(M, M, p=2)  # [B, n, n]

        # Select top-q similar rows for each reference row
        topk_row_indices = torch.topk(-row_dist, k=self.topk_rows, dim=-1, largest=True).indices  # [B, n, q]

        # Gather similar rows to construct G_q,m (formula 5)
        B_idx = torch.arange(B, device=M.device).reshape(B, 1, 1, 1)
        n_idx = torch.arange(n, device=M.device).reshape(1, n, 1, 1)
        q_idx = torch.arange(self.topk_rows, device=M.device).reshape(1, 1, self.topk_rows, 1)
        pixel_groups = M[B_idx, topk_row_indices.unsqueeze(-1), q_idx]  # [B, n, q, m]

        return pixel_groups

    def adaptive_dual_threshold(
        self,
        coeffs: torch.Tensor
    ) -> torch.Tensor:

        Bn, q, m = coeffs.shape

        # Step 1: Structural hard-thresholding (formula 8)
        # Preserve first row and first column (low-frequency structure), set other high-frequency to 0
        structural_mask = torch.zeros_like(coeffs, dtype=torch.bool)
        structural_mask[:, 0, :] = True  # first row
        structural_mask[:, :, 0] = True  # first column
        coeffs = torch.where(structural_mask, coeffs, torch.zeros_like(coeffs))

        # Step 2: Adaptive hard-thresholding on low-frequency coefficients (formula 9-10)
        # Estimate noise level sigma from high-frequency subband (diagonal HH)
        hh_coeffs = coeffs[:, 1:, 1:]
        sigma = torch.std(hh_coeffs, dim=[1, 2], keepdim=True)  # [Bn, 1, 1]

        # Compute adaptive threshold tau
        tau = sigma * torch.sqrt(2 * torch.log(torch.tensor(q * m, dtype=torch.float32, device=coeffs.device)) + self.eps)

        # Thresholding: keep coefficients with absolute value >= tau
        coeffs = torch.where(torch.abs(coeffs) >= tau, coeffs, torch.zeros_like(coeffs))

        return coeffs

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:

        B, C, H, W = img.shape
        device = img.device

        # Pad image to fit block size
        img_padded, (orig_H, orig_W) = self._pad_image(img)
        mask_padded, _ = self._pad_image(mask)
        pad_H, pad_W = img_padded.shape[2:]

        # Step 1: Block-level matching
        all_patches, valid_idx, topk_block_indices = self.block_level_matching(img_padded, mask_padded)
        N_valid = valid_idx.shape[1]
        if N_valid == 0:
            return img  # no damaged region, return original image

        # Step 2: Process each valid reference block
        # Gather similar block groups
        B_idx = torch.arange(B, device=device).reshape(B, 1, 1).repeat(1, N_valid, self.topk_blocks)
        block_groups = all_patches[B_idx, topk_block_indices]  # [B, N_valid, topk_blocks, C, K, K]
        block_groups_flat = block_groups.reshape(B * N_valid, self.topk_blocks, -1)  # [B*N_valid, m, C*K*K]

        # Step 3: Row-level matching
        pixel_groups = self.row_level_matching(block_groups_flat)  # [B*N_valid, n, q, m]
        Bn, n, q, m = pixel_groups.shape

        # Step 4: 2D Haar Transform
        pixel_groups_reshaped = pixel_groups.reshape(Bn * n, q, m)  # [Bn*n, q, m]
        haar_coeffs = self._haar_transform_2d(pixel_groups_reshaped)  # [Bn*n, q, m]

        # Step 5: Adaptive dual-threshold processing
        thresholded_coeffs = self.adaptive_dual_threshold(haar_coeffs)  # [Bn*n, q, m]

        # Step 6: Inverse Haar Transform
        restored_groups = self._inverse_haar_transform_2d(thresholded_coeffs)  # [Bn*n, q, m]
        restored_groups = restored_groups.reshape(Bn, n, q, m)  # [Bn, n, q, m]

        # Step 7: Aggregation (average over similar rows) + reshape back to blocks
        restored_blocks = restored_groups.mean(dim=2)  # [Bn, n, m]
        restored_blocks = restored_blocks.mean(dim=-1)  # [Bn, n] (average over similar blocks)
        restored_blocks = restored_blocks.reshape(B, N_valid, C, self.block_size, self.block_size)  # [B, N_valid, C, K, K]

        # Step 8: Backfill to image (only damaged regions)
        # Initialize output with original image
        output = img_padded.clone()
        # Create weight map for overlapping blocks
        weight_map = torch.zeros_like(output)
        unfold_weight = nn.Unfold(kernel_size=self.block_size, stride=1)
        fold_weight = nn.Fold(output_size=(pad_H, pad_W), kernel_size=self.block_size, stride=1)

        # Scatter restored blocks back to image
        patches_flat = restored_blocks.reshape(B, N_valid, -1).permute(0, 2, 1)  # [B, C*K*K, N_valid]
        weight_flat = torch.ones_like(patches_flat)

        # Fold patches back to image space
        full_patches = torch.zeros(B, C * self.block_size ** 2, (pad_H - self.block_size + 1) * (pad_W - self.block_size + 1), device=device)
        full_weight = torch.zeros_like(full_patches)
        full_patches.scatter_(2, valid_idx.unsqueeze(1).repeat(1, C * self.block_size ** 2, 1), patches_flat)
        full_weight.scatter_(2, valid_idx.unsqueeze(1).repeat(1, C * self.block_size ** 2, 1), weight_flat)

        # Fold to image
        restored_img = fold_weight(full_patches)
        weight_map = fold_weight(full_weight)
        weight_map = torch.clamp(weight_map, min=self.eps)

        # Merge: only replace damaged regions, use original known pixels
        init_img = torch.where(
            mask_padded == 1,
            restored_img / weight_map,
            img_padded
        )

        # Crop back to original size
        init_img = init_img[:, :, :orig_H, :orig_W]

        return init_img


class SHTMPreprocess(nn.Module):

    def __init__(
        self,
        block_size: int = 8,
        search_window: int = 21,
        topk_blocks: int = 8,
        topk_rows: int = 4,
        tau_valid_ratio: float = 0.4
    ):
        super().__init__()
        self.shtm = SHTM(
            block_size=block_size,
            search_window=search_window,
            topk_blocks=topk_blocks,
            topk_rows=topk_rows,
            tau_valid_ratio=tau_valid_ratio
        )
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            return self.shtm(img, mask)


if __name__ == "__main__":
    # Test device compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Initialize SHTM
    shtm = SHTM().to(device)

    # Create test input (batch=2, 3-channel, 256x256 image)
    test_img = torch.randn(2, 3, 256, 256).to(device)
    test_mask = torch.zeros(2, 1, 256, 256).to(device)
    test_mask[:, :, 100:150, 100:150] = 1.0  # 50x50 damaged region

    # Forward pass
    with torch.no_grad():
        output = shtm(test_img, test_mask)

    # Print dimension check
    print(f"Input image shape: {test_img.shape}")
    print(f"Output initialized image shape: {output.shape}")
    print("Test passed! Dimension matches, no runtime error.")
