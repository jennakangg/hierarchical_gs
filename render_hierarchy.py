#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import os
import torch
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_post
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from lpipsPyTorch import lpips
import numpy as np
import matplotlib.pyplot as plt
import random
from utils.dataset_utils import compute_save_vdp_heatmap, is_within_frustum

from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights

def direct_collate(x):
    return x

import torch.nn.functional as F


def plot_gaussian_centers_2d(gaussians, camera, image_name="gaussian_centers.png"):

    xyz = gaussians._xyz.cpu()
    scaling = gaussians._scaling.cpu()

    # Add a homogenous coordinate for matrix multiplication
    ones = torch.ones(xyz.shape[0], 1)
    xyz_homogenous = torch.cat([xyz, ones], dim=1)

    # Apply the projection matrix
    xyz_projected = torch.mm(xyz_homogenous, camera.full_proj_transform.cpu())

    # Convert from homogenous coordinates to 2D and extract depth
    xy_coords = xyz_projected[:, :2] / xyz_projected[:, 3:4]
    depths = xyz_projected[:, 2] / xyz_projected[:, 3]

    # Normalize screen coordinates to pixel values
    xy_coords[:, 0] = (xy_coords[:, 0] + 1) * camera.image_width / 2.0
    xy_coords[:, 1] = (xy_coords[:, 1] + 1) * camera.image_height / 2.0

    # Normalize scaling values for color mapping and adjust dot sizes based on depth
    normalized_scaling = (scaling[:, 0] - scaling[:, 0].min()) / (scaling[:, 0].max() - scaling[:, 0].min())
    colors = plt.cm.plasma(normalized_scaling)  # Use a perceptually uniform colormap
    dot_sizes = (scaling[:, 0] * 100) / (1 + depths) / 10 # Scale size inversely by depth 

    fig, ax = plt.subplots(figsize=(camera.image_width / 100, camera.image_height / 100))
    scatter = ax.scatter(xy_coords[:, 0], xy_coords[:, 1], s=dot_sizes, c=colors, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Gaussian Size')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('2D Visualization of Projected Gaussian Components')
    ax.set_xlim([0, camera.image_width])
    ax.set_ylim([0, camera.image_height])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    plt.savefig(image_name)

def plot_gaussian_size_heatmap(gaussians, camera, image_name):
    # Extract Gaussian data
    xyz = gaussians._xyz.cpu()
    scaling = gaussians._scaling.cpu()

    # Add a homogeneous coordinate for matrix multiplication
    ones = torch.ones(xyz.shape[0], 1)
    xyz_homogeneous = torch.cat([xyz, ones], dim=1)

    # Apply the projection matrix
    xyz_projected = torch.mm(xyz_homogeneous, camera.full_proj_transform.cpu())

    # Convert from homogeneous coordinates to 2D and extract depth
    w_coords = xyz_projected[:, 3:4]
    xy_coords = xyz_projected[:, :2] / xyz_projected[:, 3:4]
    depths = xyz_projected[:, 2] / xyz_projected[:, 3]

    valid_mask = torch.isfinite(xy_coords).all(dim=1) & torch.isfinite(depths) & (w_coords[:, 0] != 0)


    # Normalize screen coordinates to pixel values
    xy_coords[:, 0] = (xy_coords[:, 0] + 1) * camera.image_width / 2.0
    xy_coords[:, 1] = (xy_coords[:, 1] + 1) * camera.image_height / 2.0

    # Initialize a 2D heatmap
    heatmap = torch.zeros(camera.image_height, camera.image_width)

    # Populate the heatmap with Gaussian sizes (normalized and scaled based on depth)
    for i in range(xy_coords.shape[0]):
        x, y = int(xy_coords[i, 0]), int(xy_coords[i, 1])
        if 0 <= x < camera.image_width and 0 <= y < camera.image_height:
            size = scaling[i, 0] / (1 + depths[i])  # Adjust size by depth
            heatmap[y, x] += size  # Add size to the corresponding pixel

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap, cmap='plasma')
    ax.set_title('Gaussian Size Heatmap')
    plt.colorbar(im, ax=ax, label='Gaussian Size')

    plt.tight_layout()
    plt.savefig(image_name)


import matplotlib.pyplot as plt
import torch

def plot_rgb_heatmaps(image, gt_image, image_name):
    # Ensure both tensors are on CPU for plotting
    img = image.cpu().detach()
    gt_img = gt_image.cpu().detach()

    # Channels
    channels = ['R', 'G', 'B']
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    for i, channel in enumerate(channels):
        # Original Image
        ax = axs[i, 0]
        im = ax.imshow(img[i], cmap='viridis')
        ax.title.set_text(f'Generated {channel} Channel')
        fig.colorbar(im, ax=ax)
        
        # Ground Truth Image
        ax = axs[i, 1]
        im = ax.imshow(gt_img[i], cmap='viridis')
        ax.title.set_text(f'Ground Truth {channel} Channel')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(image_name)



@torch.no_grad()
def render_set(args, scene, pipe, taus, eval):

    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    interpolation_weights = torch.zeros(scene.gaussians._xyz.size(0)).float().cuda()
    num_siblings = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()

    psnr_test = 0.0
    ssims = 0.0
    lpipss = 0.0

    cameras = scene.getTestCameras() if eval else scene.getTrainCameras()

    for idx, viewpoint in enumerate(tqdm(cameras)):
        
        images = []
        viewpoint = viewpoint
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()


        cam_position = viewpoint.get_camera_position()
        cam_dir = viewpoint.get_camera_facing_direction()
        cam_R = viewpoint.R

        starting_pos = torch.tensor([-56.7730,   6.2273, -11.0258], dtype=torch.float64)

        starting_dir = torch.tensor([ 0.0635,  0.9939, -0.0897], dtype=torch.float64)

        starting_R = torch.tensor([[ 0.98109321,  0.18283624,  0.06345884],
                                    [-0.04566442, -0.09994473,  0.99394457],
                                    [ 0.18807147, -0.97805008, -0.08970599]])

        if not is_within_frustum(cam_position, cam_dir, cam_R, starting_pos, starting_dir, starting_R, max_angle_deg=40, 
                                    max_distance=40, max_rotation_deg=40):
            continue 

        for tau in range(taus[-1]):            
            tanfovx = math.tan(viewpoint.FoVx * 0.5)
            threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)

            to_render = expand_to_size(
                scene.gaussians.nodes,
                scene.gaussians.boxes,
                threshold,
                viewpoint.camera_center,
                torch.zeros((3)),
                render_indices,
                parent_indices,
                nodes_for_render_indices)

            indices = render_indices[:to_render].int().contiguous()
            node_indices = nodes_for_render_indices[:to_render].contiguous()

            get_interpolation_weights(
                node_indices,
                threshold,
                scene.gaussians.nodes,
                scene.gaussians.boxes,
                viewpoint.camera_center.cpu(),
                torch.zeros((3)),
                interpolation_weights,
                num_siblings
            )

            rgb_tensor = torch.tensor([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], dtype=torch.float32, device="cuda")
            black_tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
            white_tensor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

            image = render_post(
                viewpoint, 
                scene.gaussians, 
                pipe, 
                white_tensor, 
                render_indices=indices,
                parent_indices=parent_indices,
                interpolation_weights=interpolation_weights,
                num_node_kids=num_siblings, 
                use_trained_exp=args.train_test_exp,
                view_num=idx
                )["render"]
            
            

            gt_image = viewpoint.original_image.to("cuda")

            alpha_mask = viewpoint.alpha_mask.cuda()

            if args.train_test_exp:
                image = image[..., image.shape[-1] // 2:]
                gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                alpha_mask = alpha_mask[..., alpha_mask.shape[-1] // 2:]

            render_path = os.path.join(args.out_dir, f"render_{tau}")

            try:
                torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))
            except:
                os.makedirs(os.path.dirname(os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png")), exist_ok=True)
                torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))

            # plot_gaussian_centers_2d(scene.gaussians, viewpoint, image_name="gauss_points/" + viewpoint.image_name.split(".")[0] + ".png")
            # plot_gaussian_size_heatmap(scene.gaussians, viewpoint, image_name="gauss_sizes/" + viewpoint.image_name.split(".")[0] + ".png")
            images.append(image)

        compute_save_vdp_heatmap('example_lod_pairs_white_dense', 
                                    viewpoint, 
                                    images, 
                                    viewpoint.image_name.replace("/", "_"), 
                                    args.taus, 
                                    pixels_per_deg=32)
        
        images = []
        
        if eval:
            image *= alpha_mask
            gt_image *= alpha_mask
            psnr_test += psnr(image, gt_image).mean().double()
            ssims += ssim(image, gt_image).mean().double()
            lpipss += lpips(image, gt_image, net_type='vgg').mean().double()

        torch.cuda.empty_cache()
    if eval and len(scene.getTestCameras()) > 0:
        psnr_test /= len(scene.getTestCameras())
        ssims /= len(scene.getTestCameras())
        lpipss /= len(scene.getTestCameras())
        print(f"tau: {tau}, PSNR: {psnr_test:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--out_dir', type=str, default="")
    # parser.add_argument("--taus", nargs="+", type=float, default=[0.0, 3.0, 6.0, 15.0])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.0, 10.0, 20.0, 30.0, 40.0])

    args = parser.parse_args(sys.argv[1:])
    
    print("Rendering " + args.model_path)

    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
    render_set(args, scene, pipe, args.taus, args.eval)