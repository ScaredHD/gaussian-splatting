"""
Online Gaussian initialization: bootstrap + one-shot birth.

Implements the v1 spec from:
  docs/3_22/2026-03-22_稀疏初始化与在线Gaussian补点方法设计.md

Only two public entry points:
  - run_bootstrap()
  - run_birth()
"""

import json
import torch
import torch.nn.functional as F
import numpy as np

from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid

EPS = 1e-8


# ── AABB ──────────────────────────────────────────────────────────────────────

def load_aabb(path: str) -> dict:
    """Load a frozen AABB json.  Returns {"min": Tensor[3], "max": Tensor[3]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "min": torch.tensor(data["min"], dtype=torch.float32),
        "max": torch.tensor(data["max"], dtype=torch.float32),
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_res(res_str: str):
    """Parse '16 16 16' -> (16, 16, 16)."""
    return tuple(int(x) for x in res_str.strip().split())


def _build_voxel_centers(aabb, res):
    """Build a flat (N,3) tensor of voxel centres inside the AABB.

    Returns centres (N,3) and per-axis voxel sizes (3,).
    """
    bb_min = aabb["min"].float()
    bb_max = aabb["max"].float()
    sizes = (bb_max - bb_min) / torch.tensor(res, dtype=torch.float32)

    # half-voxel offset so centres sit inside the AABB
    x = torch.linspace(bb_min[0] + sizes[0] / 2, bb_max[0] - sizes[0] / 2, res[0])
    y = torch.linspace(bb_min[1] + sizes[1] / 2, bb_max[1] - sizes[1] / 2, res[1])
    z = torch.linspace(bb_min[2] + sizes[2] / 2, bb_max[2] - sizes[2] / 2, res[2])
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    centres = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (N,3)
    return centres, sizes


def _evenly_spaced_indices(count: int, target_count: int):
    """Return deterministic, increasing, approximately-even indices in [0, count)."""
    if count <= 0 or target_count <= 0:
        return []
    if count <= target_count:
        return list(range(count))

    raw = np.linspace(0, count - 1, num=target_count)
    indices = [int(round(v)) for v in raw.tolist()]
    indices = [max(0, min(count - 1, i)) for i in indices]

    dedup = []
    last = None
    for idx in indices:
        if idx != last:
            dedup.append(idx)
            last = idx

    if len(dedup) < target_count:
        used = set(dedup)
        for idx in range(count):
            if idx not in used:
                dedup.append(idx)
                used.add(idx)
            if len(dedup) == target_count:
                break
        dedup.sort()

    return dedup[:target_count]


def _project_to_ndc(points, full_proj_transform):
    """Project (N,3) world points -> (N,2) NDC in [-1,1].

    Returns grid (N,2) and valid mask (N,).
    """
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    pts_h = torch.cat([points, ones], dim=1)                     # (N,4)
    clip = pts_h @ full_proj_transform                            # (N,4)
    w = clip[:, 3]
    valid_w = torch.abs(w) > EPS

    grid = torch.zeros((points.shape[0], 2), dtype=points.dtype, device=points.device)
    if valid_w.any():
        ndc = clip[valid_w, :2] / w[valid_w].unsqueeze(-1)
        grid[valid_w, 0] = ndc[:, 0]
        grid[valid_w, 1] = -ndc[:, 1]

    inside = (
        valid_w
        & torch.isfinite(grid).all(dim=1)
        & (grid[:, 0] >= -1.0) & (grid[:, 0] <= 1.0)
        & (grid[:, 1] >= -1.0) & (grid[:, 1] <= 1.0)
    )
    return grid, inside


def _sample_image(image_chw, grid_n2):
    """Bilinear-sample a (C,H,W) image at (N,2) NDC coords -> (N,C)."""
    if grid_n2.shape[0] == 0:
        return torch.empty((0, image_chw.shape[0]), dtype=image_chw.dtype, device=image_chw.device)
    g = grid_n2.view(1, grid_n2.shape[0], 1, 2)             # (1,N,1,2)
    sampled = F.grid_sample(
        image_chw.unsqueeze(0), g, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )                                                         # (1,C,N,1)
    return sampled.squeeze(0).squeeze(-1).permute(1, 0)       # (N,C)


def _init_gaussian_params(xyz, rgb_linear, voxel_sizes, init_opacity, sh_degree):
    """Package xyz + rgb into the full parameter dict expected by append_gaussians.

    Parameters
    ----------
    xyz : (K,3) float  – world positions
    rgb_linear : (K,3) float – linear RGB in [0,1]
    voxel_sizes : (3,) float – per-axis voxel sizes
    init_opacity : float
    sh_degree : int

    Returns dict with keys: xyz, features_dc, features_rest, scaling, rotation, opacity
    """
    K = xyz.shape[0]
    device = xyz.device

    # SH DC from linear RGB
    sh_dc = RGB2SH(rgb_linear)                                         # (K,3)
    features_dc = sh_dc.unsqueeze(1).contiguous()                      # (K,1,3)

    n_rest = (sh_degree + 1) ** 2 - 1
    features_rest = torch.zeros((K, n_rest, 3), device=device)         # (K, rest, 3)

    # scale = log(0.75 * voxel_size) per axis
    scale_val = (0.75 * voxel_sizes.to(device)).log()                  # (3,)
    scaling = scale_val.unsqueeze(0).expand(K, 3).contiguous()         # (K,3)

    rotation = torch.zeros((K, 4), device=device)
    rotation[:, 0] = 1.0                                               # identity quat

    opacity = inverse_sigmoid(
        init_opacity * torch.ones((K, 1), device=device)
    )

    return {
        "xyz": xyz,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "scaling": scaling,
        "rotation": rotation,
        "opacity": opacity,
    }


# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────

def run_bootstrap(aabb, train_cameras, opt, sh_degree):
    """Generate bootstrap seed Gaussians.

    Returns a param dict (same keys as _init_gaussian_params) on CUDA,
    or None if no valid voxel is found.
    """
    res = _parse_res(opt.online_bootstrap_res)
    topk = opt.online_bootstrap_topk
    alpha_thr = opt.online_birth_support_alpha_thr   # reuse 0.05
    init_opacity = opt.online_init_opacity

    centres, voxel_sizes = _build_voxel_centers(aabb, res)           # (N,3), (3,)
    N = centres.shape[0]
    centres_cuda = centres.cuda()

    # per-voxel accumulators
    valid_count = torch.zeros(N, dtype=torch.int32, device="cuda")   # |V(x)|
    occ_sum = torch.zeros(N, dtype=torch.float32, device="cuda")

    # colour accumulators  (alpha-weighted straight RGB)
    colour_weight_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    colour_rgb_sum = torch.zeros((N, 3), dtype=torch.float32, device="cuda")

    for cam in train_cameras:
        grid, inside = _project_to_ndc(centres_cuda, cam.full_proj_transform)
        if not inside.any():
            continue

        gt_alpha = cam.alpha_mask.cuda()                              # (1,H,W)
        gt_rgb = cam.original_image.cuda()                            # (3,H,W)

        alpha_vals = _sample_image(gt_alpha, grid)                    # (N,1)
        alpha_vals = alpha_vals.squeeze(-1).clamp(0.0, 1.0)           # (N,)
        rgb_vals = _sample_image(gt_rgb, grid)                        # (N,3)
        rgb_vals = rgb_vals.clamp(0.0, 1.0)

        valid_count[inside] += 1
        occ_sum[inside] += alpha_vals[inside]

        # colour aggregation: only views where alpha > thr
        support_mask = inside & (alpha_vals > alpha_thr)
        if support_mask.any():
            w = alpha_vals[support_mask]
            colour_weight_sum[support_mask] += w
            colour_rgb_sum[support_mask] += w.unsqueeze(-1) * rgb_vals[support_mask]

    # filter: |V(x)| >= 8
    valid_mask = valid_count >= 8
    if not valid_mask.any():
        print("[bootstrap] WARNING: no voxel has >= 8 valid views. Returning None.")
        return None

    occ = occ_sum.clone()
    occ[valid_mask] /= valid_count[valid_mask].float()
    occ[~valid_mask] = 0.0

    # top-K by occ among valid voxels
    scores = occ.clone()
    scores[~valid_mask] = -1.0
    k = min(topk, int(valid_mask.sum().item()))
    _, topk_idx = torch.topk(scores, k)

    sel_centres = centres_cuda[topk_idx]
    sel_weights = colour_weight_sum[topk_idx]
    sel_rgb_sum = colour_rgb_sum[topk_idx]
    sel_rgb = sel_rgb_sum / (sel_weights.unsqueeze(-1) + EPS)
    sel_rgb = sel_rgb.clamp(0.0, 1.0)

    print(f"[bootstrap] selected {k} seeds from {int(valid_mask.sum().item())} valid voxels "
          f"(grid {res[0]}x{res[1]}x{res[2]}, total {N})")

    return _init_gaussian_params(sel_centres, sel_rgb, voxel_sizes, init_opacity, sh_degree)


# ── ONE-SHOT BIRTH ────────────────────────────────────────────────────────────

def run_birth(aabb, train_cameras, gaussians, render_func, pipe, background, separate_sh, opt, sh_degree):
    """One-shot birth at a fixed iteration.

    Returns a param dict on CUDA, or None if no candidate survives.
    """
    res = _parse_res(opt.online_birth_res)
    topk = opt.online_birth_topk
    n_views = opt.online_birth_views
    min_valid = opt.online_birth_valid_views_min
    alpha_thr = opt.online_birth_support_alpha_thr
    ratio_min = opt.online_birth_support_ratio_min
    repel_mult = opt.online_birth_repel_radius_mult
    init_opacity = opt.online_init_opacity

    # deterministic view sampling: sort by image_name then pick evenly spaced views
    sorted_cams = sorted(train_cameras, key=lambda cam: cam.image_name)
    total_views = len(sorted_cams)
    view_indices = _evenly_spaced_indices(total_views, n_views)
    selected_cams = [sorted_cams[i] for i in view_indices]

    print(f"[birth] using {len(selected_cams)} views (sorted indices {view_indices}) "
          f"from {total_views} train views")

    centres, voxel_sizes = _build_voxel_centers(aabb, res)
    N = centres.shape[0]
    centres_cuda = centres.cuda()
    min_voxel = voxel_sizes.min().item()

    # per-voxel accumulators
    valid_count = torch.zeros(N, dtype=torch.int32, device="cuda")
    q_sum = torch.zeros(N, dtype=torch.float32, device="cuda")        # sum m_t * r_t
    support_count = torch.zeros(N, dtype=torch.int32, device="cuda")  # sum m_t

    # colour accumulators
    colour_weight_sum = torch.zeros(N, dtype=torch.float32, device="cuda")
    colour_rgb_sum = torch.zeros((N, 3), dtype=torch.float32, device="cuda")

    for cam in selected_cams:
        # render current prediction
        with torch.no_grad():
            render_pkg = render_func(cam, gaussians, pipe, background, separate_sh=separate_sh)
        pred_alpha = render_pkg["alpha"].detach()                      # (1,H,W)
        gt_alpha = cam.alpha_mask.cuda()                               # (1,H,W)
        gt_rgb = cam.original_image.cuda()                             # (3,H,W)

        grid, inside = _project_to_ndc(centres_cuda, cam.full_proj_transform)
        if not inside.any():
            continue

        a_gt = _sample_image(gt_alpha, grid).squeeze(-1).clamp(0.0, 1.0)    # (N,)
        a_pred = _sample_image(pred_alpha, grid).squeeze(-1).clamp(0.0, 1.0)
        rgb_vals = _sample_image(gt_rgb, grid).clamp(0.0, 1.0)              # (N,3)

        r_t = (a_gt - a_pred).clamp(min=0.0)                                # (N,)
        m_t = (inside & (a_gt > alpha_thr)).float()                          # (N,)

        valid_count[inside] += 1
        q_sum[inside] += (m_t * r_t)[inside]
        support_count += m_t.int()

        sup_mask = inside & (m_t > 0.5)
        if sup_mask.any():
            w = a_gt[sup_mask]
            colour_weight_sum[sup_mask] += w
            colour_rgb_sum[sup_mask] += w.unsqueeze(-1) * rgb_vals[sup_mask]

    # ── gating ────────────────────────────────────────────────────────────────
    q = q_sum.clone()
    valid_mask = valid_count >= min_valid
    q[valid_mask] /= valid_count[valid_mask].float()
    q[~valid_mask] = 0.0

    support_ratio = torch.zeros(N, dtype=torch.float32, device="cuda")
    support_ratio[valid_mask] = support_count[valid_mask].float() / valid_count[valid_mask].float()

    gate_mask = valid_mask & (support_ratio >= ratio_min) & (q > 0.0)

    print(f"[birth] valid voxels: {int(valid_mask.sum())}, "
          f"gate pass (before repulsion): {int(gate_mask.sum())}")

    if not gate_mask.any():
        print("[birth] no candidate after gating. Returning None.")
        return None

    # ── repulsion ─────────────────────────────────────────────────────────────
    existing_xyz = gaussians.get_xyz.detach()  # (M,3)
    if existing_xyz.shape[0] > 0:
        cand_idx = gate_mask.nonzero(as_tuple=True)[0]
        cand_pts = centres_cuda[cand_idx]      # (C,3)

        # brute-force nearest-neighbour (C small enough for 64^3 scenario)
        # chunk to avoid OOM if candidate count is huge
        repel_dist = repel_mult * min_voxel
        keep = torch.ones(cand_idx.shape[0], dtype=torch.bool, device="cuda")

        chunk_size = 4096
        for start in range(0, cand_pts.shape[0], chunk_size):
            end = min(start + chunk_size, cand_pts.shape[0])
            dists = torch.cdist(cand_pts[start:end], existing_xyz)   # (chunk, M)
            d_min, _ = dists.min(dim=1)
            keep[start:end] = d_min >= repel_dist

        rejected = int((~keep).sum().item())
        gate_mask[cand_idx] = keep

        print(f"[birth] repulsion rejected {rejected} candidates (radius {repel_dist:.4f})")

    if not gate_mask.any():
        print("[birth] no candidate after repulsion. Returning None.")
        return None

    # ── top-K ─────────────────────────────────────────────────────────────────
    cand_idx = gate_mask.nonzero(as_tuple=True)[0]
    cand_scores = q[cand_idx]
    if cand_idx.numel() == 0:
        print("[birth] no candidate for ranking. Returning None.")
        return None

    # Enforce tie-break rule exactly: score desc, then linear voxel index asc.
    index_order = torch.argsort(cand_idx)
    cand_idx = cand_idx[index_order]
    cand_scores = cand_scores[index_order]
    score_order = torch.argsort(cand_scores, descending=True, stable=True)
    ranked_idx = cand_idx[score_order]

    k = min(topk, int(ranked_idx.shape[0]))
    topk_idx = ranked_idx[:k]

    sel_centres = centres_cuda[topk_idx]
    sel_weights = colour_weight_sum[topk_idx]
    sel_rgb_sum = colour_rgb_sum[topk_idx]
    sel_rgb = sel_rgb_sum / (sel_weights.unsqueeze(-1) + EPS)
    sel_rgb = sel_rgb.clamp(0.0, 1.0)

    print(f"[birth] selected {k} new Gaussians (from {int(gate_mask.sum())} candidates, "
          f"grid {res[0]}x{res[1]}x{res[2]})")

    return _init_gaussian_params(sel_centres, sel_rgb, voxel_sizes, init_opacity, sh_degree)
