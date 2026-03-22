# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision


EPS = 1e-8
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def get_view_targets(view):
    gt_rgb = view.original_image[0:3, :, :]
    if view.alpha_mask is not None:
        gt_alpha = view.alpha_mask[0:1, :, :]
    else:
        gt_alpha = torch.ones_like(gt_rgb[0:1, :, :])
    gt_premul = gt_rgb * gt_alpha
    return gt_rgb, gt_alpha, gt_premul



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, save_alpha_outputs):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    alpha_pred_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha_pred")
    alpha_gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha_gt")
    gt_premul_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_premul")
    render_straight_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_straight")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if save_alpha_outputs:
        makedirs(alpha_pred_path, exist_ok=True)
        makedirs(alpha_gt_path, exist_ok=True)
        makedirs(gt_premul_path, exist_ok=True)
        makedirs(render_straight_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_pkg["render"]
        pred_alpha = render_pkg["alpha"]
        gt_rgb, gt_alpha, gt_premul = get_view_targets(view)

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2 :]
            pred_alpha = pred_alpha[..., pred_alpha.shape[-1] // 2 :]
            gt_rgb = gt_rgb[..., gt_rgb.shape[-1] // 2 :]
            gt_alpha = gt_alpha[..., gt_alpha.shape[-1] // 2 :]
            gt_premul = gt_premul[..., gt_premul.shape[-1] // 2 :]

        filename = "{0:05d}.png".format(idx)
        torchvision.utils.save_image(rendering, os.path.join(render_path, filename))
        torchvision.utils.save_image(gt_rgb, os.path.join(gts_path, filename))
        if save_alpha_outputs:
            pred_rgb_straight = torch.where(pred_alpha > EPS, rendering / pred_alpha.clamp_min(EPS), torch.zeros_like(rendering)).clamp(0.0, 1.0)
            torchvision.utils.save_image(pred_alpha, os.path.join(alpha_pred_path, filename))
            torchvision.utils.save_image(gt_alpha, os.path.join(alpha_gt_path, filename))
            torchvision.utils.save_image(gt_premul, os.path.join(gt_premul_path, filename))
            torchvision.utils.save_image(pred_rgb_straight, os.path.join(render_straight_path, filename))



def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, separate_sh: bool, save_alpha_outputs: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, save_alpha_outputs)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, save_alpha_outputs)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_alpha_outputs", action="store_true", default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.save_alpha_outputs)
