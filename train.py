# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from online_init import load_aabb, run_bootstrap, run_birth
from scene import GaussianModel, Scene
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


EPS = 1e-8
SUPPORT_DIAGNOSTIC_THRESHOLD = 0.10
SUPPORT_PRUNE_MIN_HITS = 8
SUPPORT_PRUNE_OPACITY_CAP = 0.02
SUPPORT_BASE_RADIUS_PX = 2.0
SUPPORT_INNER_OFFSETS_PX = (
    (0.0, 0.0),
    (0.5 * SUPPORT_BASE_RADIUS_PX, 0.0),
    (-0.5 * SUPPORT_BASE_RADIUS_PX, 0.0),
    (0.0, 0.5 * SUPPORT_BASE_RADIUS_PX),
    (0.0, -0.5 * SUPPORT_BASE_RADIUS_PX),
)
SUPPORT_RING_OFFSETS_PX = (
    (1.5 * SUPPORT_BASE_RADIUS_PX, 0.0),
    (-1.5 * SUPPORT_BASE_RADIUS_PX, 0.0),
    (0.0, 1.5 * SUPPORT_BASE_RADIUS_PX),
    (0.0, -1.5 * SUPPORT_BASE_RADIUS_PX),
)


def get_ground_truth_tensors(viewpoint, alpha_bg_threshold=None, alpha_fg_threshold=None):
    gt_rgb = viewpoint.original_image.cuda()
    if viewpoint.alpha_mask is not None:
        gt_alpha = viewpoint.alpha_mask.cuda()
    else:
        gt_alpha = torch.ones_like(gt_rgb[:1, ...])

    gt_premul = gt_rgb * gt_alpha
    boundary_mask = None
    if alpha_bg_threshold is not None and alpha_fg_threshold is not None:
        boundary_mask = ((gt_alpha > alpha_bg_threshold) & (gt_alpha < alpha_fg_threshold)).float()

    return gt_rgb, gt_alpha, gt_premul, boundary_mask



def apply_train_test_crop(train_test_exp, *tensors):
    if not train_test_exp:
        return tensors
    cropped = []
    for tensor in tensors:
        cropped.append(tensor[..., tensor.shape[-1] // 2 :])
    return tuple(cropped)



def compute_rgb_loss(pred_rgb, gt_premul, lambda_dssim):
    rgb_l1 = l1_loss(pred_rgb, gt_premul)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(pred_rgb.unsqueeze(0), gt_premul.unsqueeze(0))
    else:
        ssim_value = ssim(pred_rgb, gt_premul)
    rgb_loss = (1.0 - lambda_dssim) * rgb_l1 + lambda_dssim * (1.0 - ssim_value)
    return rgb_loss, rgb_l1, ssim_value



def compute_weighted_alpha_l1(pred_alpha, gt_alpha, boundary_mask, boundary_alpha_weight):
    weights = torch.ones_like(gt_alpha)
    if boundary_mask is not None:
        weights = weights + boundary_alpha_weight * boundary_mask
    return (weights * torch.abs(pred_alpha - gt_alpha)).sum() / (weights.sum() + EPS)


def flatten_visibility_filter(visibility_filter):
    return visibility_filter.long().view(-1)


def project_points_to_image_grid(points, full_proj_transform):
    if points.shape[0] == 0:
        empty_grid = torch.empty((0, 2), dtype=points.dtype, device=points.device)
        empty_mask = torch.empty((0,), dtype=torch.bool, device=points.device)
        return empty_grid, empty_mask

    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat((points, ones), dim=1)
    clip_points = points_h @ full_proj_transform
    w = clip_points[:, 3]
    valid_w = torch.abs(w) > EPS

    grid = torch.zeros((points.shape[0], 2), dtype=points.dtype, device=points.device)
    if valid_w.any():
        ndc = clip_points[valid_w, :2] / w[valid_w].unsqueeze(-1)
        grid[valid_w, 0] = ndc[:, 0]
        grid[valid_w, 1] = -ndc[:, 1]

    inside_mask = valid_w & torch.isfinite(grid).all(dim=1)
    inside_mask = inside_mask & (grid[:, 0] >= -1.0) & (grid[:, 0] <= 1.0) & (grid[:, 1] >= -1.0) & (grid[:, 1] <= 1.0)
    return grid, inside_mask


def sample_alpha_at_grid(gt_alpha, sample_grid):
    if sample_grid.shape[0] == 0:
        return torch.empty((0,), dtype=gt_alpha.dtype, device=gt_alpha.device)

    grid = sample_grid.view(1, sample_grid.shape[0], 1, 2)
    sampled = F.grid_sample(gt_alpha.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.view(-1).clamp(0.0, 1.0)


def sample_alpha_with_offsets(gt_alpha, base_grid, image_width, image_height, offsets_px):
    if base_grid.shape[0] == 0:
        return torch.empty((0, len(offsets_px)), dtype=gt_alpha.dtype, device=gt_alpha.device)

    grid_offsets = []
    for offset_x_px, offset_y_px in offsets_px:
        offset_grid = base_grid.clone()
        if image_width > 1:
            offset_grid[:, 0] += (2.0 * offset_x_px) / float(image_width - 1)
        if image_height > 1:
            offset_grid[:, 1] += (2.0 * offset_y_px) / float(image_height - 1)
        grid_offsets.append(offset_grid)

    stacked_grid = torch.stack(grid_offsets, dim=1).reshape(-1, 2)
    sampled = sample_alpha_at_grid(gt_alpha, stacked_grid)
    return sampled.view(base_grid.shape[0], len(offsets_px))


def compute_visible_support_observations(viewpoint, gaussians, gt_alpha, visibility_filter):
    visible_indices = flatten_visibility_filter(visibility_filter)
    if visible_indices.numel() == 0:
        empty_indices = torch.empty((0,), dtype=torch.long, device=gt_alpha.device)
        empty_values = torch.empty((0,), dtype=gt_alpha.dtype, device=gt_alpha.device)
        return empty_indices, empty_values, empty_values

    visible_points = gaussians.get_xyz[visible_indices]
    sample_grid, update_mask = project_points_to_image_grid(visible_points, viewpoint.full_proj_transform)
    if not update_mask.any():
        empty_indices = torch.empty((0,), dtype=torch.long, device=gt_alpha.device)
        empty_values = torch.empty((0,), dtype=gt_alpha.dtype, device=gt_alpha.device)
        return empty_indices, empty_values, empty_values

    updated_indices = visible_indices[update_mask]
    updated_grid = sample_grid[update_mask]
    a_in = sample_alpha_with_offsets(
        gt_alpha,
        updated_grid,
        viewpoint.image_width,
        viewpoint.image_height,
        SUPPORT_INNER_OFFSETS_PX,
    ).mean(dim=1)
    a_ring = sample_alpha_with_offsets(
        gt_alpha,
        updated_grid,
        viewpoint.image_width,
        viewpoint.image_height,
        SUPPORT_RING_OFFSETS_PX,
    ).mean(dim=1)
    grow_observations = a_in.clamp(0.0, 1.0)
    prune_observations = (a_in * a_ring).clamp(0.0, 1.0)
    return updated_indices, grow_observations, prune_observations


def persist_support_stats(model_path, iteration, gaussians, snapshot_dir=None):
    stats = {"iteration": int(iteration)}
    stats.update(gaussians.compute_support_stats(SUPPORT_DIAGNOSTIC_THRESHOLD))

    stats_path = os.path.join(model_path, "support_stats.json")
    payload = {
        "support_mode": "dual_support_v1",
        "grow_observation": "a_in",
        "prune_observation": "a_in_times_a_ring",
        "diagnostic_threshold": SUPPORT_DIAGNOSTIC_THRESHOLD,
        "latest": stats,
        "snapshots": [stats],
    }
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        snapshots = [snap for snap in payload.get("snapshots", []) if snap.get("iteration") != int(iteration)]
        snapshots.append(stats)
        snapshots.sort(key=lambda item: item.get("iteration", -1))
        payload["snapshots"] = snapshots
        payload["latest"] = stats
        payload["support_mode"] = "dual_support_v1"
        payload["grow_observation"] = "a_in"
        payload["prune_observation"] = "a_in_times_a_ring"
        payload["diagnostic_threshold"] = SUPPORT_DIAGNOSTIC_THRESHOLD

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
        with open(os.path.join(snapshot_dir, "support_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    online_mode = (opt.online_init_mode or "").strip()
    valid_online_modes = {"", "bootstrap_only", "bootstrap_birth"}
    if online_mode not in valid_online_modes:
        valid_text = "'' (disabled), 'bootstrap_only', 'bootstrap_birth'"
        sys.exit(f"[online_init] invalid --online_init_mode '{online_mode}'. "
                 f"Valid values are: {valid_text}.")
    scene = Scene(dataset, gaussians, online_init_mode=online_mode)

    # ── online bootstrap initialization ──────────────────────────────────
    if online_mode in ("bootstrap_only", "bootstrap_birth"):
        aabb_path = opt.online_aabb_path
        if not aabb_path:
            sys.exit("[online_init] --online_aabb_path is required when online_init_mode is set.")
        if not os.path.isfile(aabb_path):
            sys.exit(f"[online_init] AABB json not found: {aabb_path}")
        aabb = load_aabb(aabb_path)
        print(f"[online_init] loaded AABB from {aabb_path}: "
              f"min={aabb['min'].tolist()}, max={aabb['max'].tolist()}")
        train_cams = scene.getTrainCameras()
        bootstrap_params = run_bootstrap(aabb, train_cams, opt, dataset.sh_degree)
        if bootstrap_params is None:
            sys.exit("[online_init] bootstrap produced no seeds. Aborting.")
        cam_infos = scene._scene_info.train_cameras + scene._scene_info.test_cameras
        gaussians.create_from_bootstrap(bootstrap_params, cam_infos, scene.cameras_extent)

    gaussians.training_setup(opt)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_total_loss_for_log = 0.0
    ema_rgb_loss_for_log = 0.0
    ema_alpha_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifier, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.increment_age()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]
        pred_alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_rgb, gt_alpha, gt_premul, boundary_mask = get_ground_truth_tensors(
            viewpoint_cam,
            alpha_bg_threshold=opt.alpha_bg_threshold,
            alpha_fg_threshold=opt.alpha_fg_threshold,
        )

        rgb_loss, rgb_l1, _ = compute_rgb_loss(image, gt_premul, opt.lambda_dssim)
        alpha_loss = compute_weighted_alpha_l1(pred_alpha, gt_alpha, boundary_mask, opt.boundary_alpha_weight)
        loss = rgb_loss + opt.lambda_alpha * alpha_loss

        ll1depth_value = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            inv_depth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            ll1depth_pure = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
            ll1depth = depth_l1_weight(iteration) * ll1depth_pure
            loss += ll1depth
            ll1depth_value = ll1depth.item()

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            support_indices, grow_support_observations, prune_support_observations = compute_visible_support_observations(
                viewpoint_cam,
                gaussians,
                gt_alpha,
                visibility_filter,
            )
            gaussians.update_support(
                support_indices,
                grow_support_observations,
                prune_support_observations,
                opt.support_beta,
            )

            ema_total_loss_for_log = 0.4 * loss.item() + 0.6 * ema_total_loss_for_log
            ema_rgb_loss_for_log = 0.4 * rgb_loss.item() + 0.6 * ema_rgb_loss_for_log
            ema_alpha_loss_for_log = 0.4 * alpha_loss.item() + 0.6 * ema_alpha_loss_for_log
            ema_depth_loss_for_log = 0.4 * ll1depth_value + 0.6 * ema_depth_loss_for_log
            support_stats = gaussians.compute_support_stats(SUPPORT_DIAGNOSTIC_THRESHOLD)

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_total_loss_for_log:.7f}",
                        "RGB": f"{ema_rgb_loss_for_log:.7f}",
                        "Alpha": f"{ema_alpha_loss_for_log:.7f}",
                        "Depth": f"{ema_depth_loss_for_log:.7f}",
                        "Grow": f"{support_stats['grow_support_mean']:.4f}",
                        "Prune": f"{support_stats['prune_support_mean']:.4f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                rgb_loss,
                alpha_loss,
                loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp,
                opt,
            )
            if iteration in testing_iterations or iteration == opt.iterations:
                persist_support_stats(dataset.model_path, iteration, gaussians)
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                persist_support_stats(
                    dataset.model_path,
                    iteration,
                    gaussians,
                    snapshot_dir=os.path.join(dataset.model_path, "point_cloud", "iteration_{}".format(iteration)),
                )

            # ── one-shot birth ────────────────────────────────────────────
            if (online_mode == "bootstrap_birth"
                    and iteration == opt.online_birth_iter):
                print(f"\n[ITER {iteration}] Running one-shot birth...")
                birth_params = run_birth(
                    aabb, scene.getTrainCameras(), gaussians,
                    render, pipe, background, SPARSE_ADAM_AVAILABLE, opt, dataset.sh_degree,
                )
                if birth_params is not None:
                    n_before = gaussians.get_xyz.shape[0]
                    gaussians.append_gaussians(birth_params)
                    n_after = gaussians.get_xyz.shape[0]
                    print(f"[birth] appended {n_after - n_before} new Gaussians "
                          f"({n_before} -> {n_after})")
                else:
                    print("[birth] no candidates survived. Continuing without new points.")

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        support_tau_densify=opt.support_tau_densify,
                        support_tau_prune=opt.support_tau_prune,
                        support_min_hits=SUPPORT_PRUNE_MIN_HITS,
                        support_min_age=2 * opt.densification_interval,
                        support_opacity_cap=SUPPORT_PRUNE_OPACITY_CAP,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



def training_report(tb_writer, iteration, rgb_loss, alpha_loss, total_loss, elapsed, testing_iterations, scene, render_func, render_args, train_test_exp, opt):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/rgb_loss", rgb_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/alpha_loss", alpha_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", total_loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
    )

    for config in validation_configs:
        if not config["cameras"] or len(config["cameras"]) == 0:
            continue

        l1_test = 0.0
        psnr_test = 0.0
        alpha_mae_test = 0.0
        for idx, viewpoint in enumerate(config["cameras"]):
            render_pkg = render_func(viewpoint, scene.gaussians, *render_args)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            pred_alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0)
            _, gt_alpha, gt_premul, _ = get_ground_truth_tensors(
                viewpoint,
                alpha_bg_threshold=opt.alpha_bg_threshold,
                alpha_fg_threshold=opt.alpha_fg_threshold,
            )
            if train_test_exp:
                image, pred_alpha, gt_premul, gt_alpha = apply_train_test_crop(train_test_exp, image, pred_alpha, gt_premul, gt_alpha)
            if tb_writer and idx < 5:
                tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(config["name"] + "_view_{}/ground_truth_premul".format(viewpoint.image_name), gt_premul[None], global_step=iteration)
            l1_test += l1_loss(image, gt_premul).mean().double()
            psnr_test += psnr(image, gt_premul).mean().double()
            alpha_mae_test += torch.abs(pred_alpha - gt_alpha).mean().double()

        psnr_test /= len(config["cameras"])
        l1_test /= len(config["cameras"])
        alpha_mae_test /= len(config["cameras"])
        print(
            "\n[ITER {}] Evaluating {}: L1Premul {} PSNRPremul {} AlphaMAE {}".format(
                iteration, config["name"], l1_test, psnr_test, alpha_mae_test
            )
        )
        if tb_writer:
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_premul", l1_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr_premul", psnr_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - alpha_mae", alpha_mae_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        support_stats = scene.gaussians.compute_support_stats(SUPPORT_DIAGNOSTIC_THRESHOLD)
        tb_writer.add_scalar("scene/grow_support_mean", support_stats["grow_support_mean"], iteration)
        tb_writer.add_scalar("scene/grow_support_median", support_stats["grow_support_median"], iteration)
        tb_writer.add_scalar("scene/prune_support_mean", support_stats["prune_support_mean"], iteration)
        tb_writer.add_scalar("scene/prune_support_median", support_stats["prune_support_median"], iteration)
        tb_writer.add_scalar("scene/ratio_prune_support_below_0_10", support_stats["ratio_prune_support_below_0_10"], iteration)
        tb_writer.add_scalar("scene/support_hits_mean", support_stats["support_hits_mean"], iteration)
        tb_writer.add_scalar("scene/support_pruned_count", support_stats["support_pruned_count"], iteration)
        tb_writer.add_scalar(
            "scene/densify_blocked_by_grow_support_count",
            support_stats["densify_blocked_by_grow_support_count"],
            iteration,
        )
        tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
