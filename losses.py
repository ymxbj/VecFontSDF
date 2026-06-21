"""Pseudo distance field and reconstruction losses.

Conventions:
- Query points live in [-1, 1] x [-1, 1]; the last dim is (x, y) = (col, row).
- g > 0 means the point lies outside the shape; g < 0 means inside.
- gamma_rasterize maps SDF values inside [-gamma, gamma] to [0, 1] via a cubic
  smoothstep so the rasterization is differentiable. Output 0 = inside the
  shape (black), 1 = outside (white), matching the training convention
  (white background, black glyph; PIL+ToTensor yields glyph pixels ~ 0).
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def gamma_rasterize(g: torch.Tensor, gamma: float) -> torch.Tensor:
    """Cubic smoothstep that maps an SDF value to a soft {0, 1} mask."""
    ratio = g / gamma
    middle = 0.5 - 0.25 * (ratio ** 3 - 3 * ratio)
    out = torch.where(g >= gamma, torch.ones_like(g),
          torch.where(g <= -gamma, torch.zeros_like(g), middle))
    return out


def pseudo_distance(params: torch.Tensor, points: torch.Tensor,
                    v_dim: int, p_dim: int) -> torch.Tensor:
    """Evaluate the pseudo distance field G at a batch of query points.

    Each parabolic curve is a 6-tuple (k, p, q, d, e, f) defining the implicit
    function H = k(px + qy)^2 + dx + ey + f. p_dim curves combine by max
    (intersection) into one convex primitive; v_dim primitives combine by min
    (union) into the final shape.

    Args:
        params: [B, v_dim*p_dim, 6]
        points: [B, N, 2]   query points (x, y) in [-1, 1]
        v_dim:  number of primitives
        p_dim:  number of curves per primitive

    Returns:
        g: [B, N]
    """
    k, p, q, d, e, f = params.split(1, dim=2)          # each [B, v*a, 1]
    x = points[:, :, 0].unsqueeze(1)                    # [B, 1, N]
    y = points[:, :, 1].unsqueeze(1)                    # [B, 1, N]
    h = k * (p * x + q * y) ** 2 + d * x + e * y + f   # [B, v*a, N]
    h = h.permute(0, 2, 1)                              # [B, N, v*a]
    h = h.view(h.size(0), h.size(1), v_dim, p_dim)
    f_per_primitive = h.amax(dim=3)                     # [B, N, v_dim]
    g = f_per_primitive.amin(dim=2)                     # [B, N]
    return g


def build_grid(image_size: int, device) -> torch.Tensor:
    """Pixel-center grid flattened in row-major order.

    Returns [H*W, 2]; the last dim is (x, y) = (row_norm, col_norm), both in
    [-1, 1]. Row-major flat_idx = row * W + col matches the layout of a PIL +
    ToTensor image after .view(-1). The (x=row, y=col) convention matches the
    released checkpoints; do not swap it or reconstructions come out transposed.
    """
    rng = torch.arange(image_size, dtype=torch.float32, device=device)
    rows, cols = torch.meshgrid(rng, rng, indexing='ij')   # rows = x, cols = y
    xs = (rows + 0.5) / (image_size / 2) - 1.0
    ys = (cols + 0.5) / (image_size / 2) - 1.0
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def compute_losses(params: torch.Tensor, image: torch.Tensor,
                   grid_sdf: torch.Tensor, contour_sdf: torch.Tensor,
                   opts, grid_coords: torch.Tensor
                   ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Compute the full reconstruction loss in a single forward pass.

    Args:
        params:       [B, v*a, 6]
        image:        [B, 1, H, W]  grayscale raster, [0, 1]
        grid_sdf:     [B, H, W]     ground-truth SDF in pixel units
        contour_sdf:  [B, M_c, 3]   each row = (x, y, signed_distance) in pixel
                                    units (x = column, y = row, both in [0, H];
                                    signed_distance > 0 outside the glyph)
        opts:         output of options.get_recon_parser()
        grid_coords:  [H*W, 2] from build_grid()

    Returns:
        (losses_dict, predicted_image)
    """
    B = image.size(0)
    H = opts.image_size

    # Merge grid + contour points into a single batched forward.
    grid_pts = grid_coords.unsqueeze(0).expand(B, -1, -1)     # [B, H*W, 2]
    # contour rows are (col, row, sdf); build_grid uses (x=row, y=col), so map
    # x <- row coordinate (col index 1), y <- col coordinate (col index 0).
    contour_xy = torch.stack([
        contour_sdf[..., 1] / (H / 2) - 1.0,                   # x = row
        contour_sdf[..., 0] / (H / 2) - 1.0,                   # y = col
    ], dim=-1)                                                 # [B, M_c, 2]
    points = torch.cat([grid_pts, contour_xy], dim=1)          # [B, H*W + M_c, 2]
    g = pseudo_distance(params, points, opts.v_dim, opts.p_dim)
    g_grid, g_contour = g[:, :H * H], g[:, H * H:]             # [B, H*W], [B, M_c]

    # Image reconstruction loss. Both prediction and target are grayscale.
    img_pred = gamma_rasterize(g_grid, opts.gamma).view(B, 1, H, H)
    img_loss = F.mse_loss(img_pred, image)

    # Grid SDF loss: penalize sign disagreement between predicted and GT SDF
    # at every pixel center.
    gt_grid = grid_sdf.reshape(B, -1) / (H / 2)                # rescale to [-1, 1] units
    grid_loss = F.relu(-g_grid * gt_grid).mean()

    # Contour SDF loss: same idea, but on points sampled near the glyph contour.
    gt_contour = contour_sdf[..., 2] / (H / 2)
    contour_loss = F.relu(-g_contour * gt_contour).mean()

    # Regularization: pq term keeps the parabola axis (p, q) unit-norm; k^2
    # term pushes mean(k^2) above k2_target so primitives do not collapse to
    # linear half-spaces.
    k = params[..., 0]
    p = params[..., 1]
    q = params[..., 2]
    pq_reg = ((p * p + q * q - 1) ** 2).mean()
    k2_mean = (k * k).mean()
    k2_violation = F.relu(opts.k2_target - k2_mean)
    regular_loss = pq_reg + opts.w_k2 * k2_violation

    total = (opts.w_image * img_loss
             + opts.w_grid * grid_loss
             + opts.w_contour * contour_loss
             + opts.w_regular * regular_loss)

    losses = {
        'image':   img_loss,
        'grid':    grid_loss,
        'contour': contour_loss,
        'regular': regular_loss,
        'k2_mean': k2_mean.detach(),
        'total':   total,
    }
    return losses, img_pred
