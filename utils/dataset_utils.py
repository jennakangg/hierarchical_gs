import torch
import numpy as np
import os
import pandas as pd
from FovVideoVDP.pyfvvdp import fvvdp
from dataset.lod_pairs_fvvdp import LODPairsFvvdp

def normalize_camera_position_to_unit_norm(camera_positions):

    norms = torch.norm(camera_positions)
    normalized_positions = camera_positions / norms
    return normalized_positions

def is_within_frustum(current_position, current_direction, current_rotation, start_position, start_direction, start_rotation, max_angle_deg, max_distance, max_rotation_deg):
    cosine_angle = np.dot(current_direction, start_direction) / (np.linalg.norm(current_direction) * np.linalg.norm(start_direction))
    angle_deg = np.arccos(np.clip(cosine_angle, -1, 1)) * (180 / np.pi)
    distance = np.linalg.norm(current_position - start_position)

    rotation_rel = np.dot(start_rotation.T, current_rotation)
    rotation_angle = np.arccos((np.trace(rotation_rel) - 1) / 2) * (180 / np.pi)

    return angle_deg <= max_angle_deg and distance <= max_distance and rotation_angle <= max_rotation_deg


def compute_save_vdp_heatmap(dataset_dir, camera, lod_tensors, image_name, levels, pixels_per_deg=32):
    os.makedirs(dataset_dir, exist_ok=True)
    dataset = LODPairsFvvdp(dataset_dir)

    level_n = levels-1

    filename_lod_n = f"cam_{image_name}_lodn{level_n}.pt"
    file_path_lod_n = os.path.join(dataset_dir, filename_lod_n)
    print(image_name)
    I_ref = np.transpose(lod_tensors[level_n].cpu().numpy(), (1, 2, 0))

    torch.save(I_ref, file_path_lod_n)

    for level_x in range(levels):
        cam_position = camera.get_camera_position()
        cam_dir = camera.get_camera_facing_direction()

        cam_R = camera.R
        cam_T = camera.T

        I_compare = np.transpose(lod_tensors[level_x].cpu().numpy(), (1, 2, 0))

        fv = fvvdp(display_name='custom', heatmap='raw', foveated=True)

        _, image_stats = fv.predict( I_compare, I_ref, dim_order="HWC")

        filename = f"cam_{image_name}_lodx{level_x}_lodn{level_n}.pt"
        file_path = os.path.join(dataset_dir, filename)
        heatmap = image_stats['heatmap']
        torch.save(heatmap, file_path)

        filename_lod_x = f"cam_{image_name}_lodn{level_x}.pt"
        file_path_lod_x = os.path.join(dataset_dir, filename_lod_x)
        torch.save(I_compare, file_path_lod_x)
        
        filename_camera_transform = f"cam_{image_name}_cam_proj_transform.pt"
        file_path_camera_transform = os.path.join(dataset_dir, filename_camera_transform)
        torch.save(camera.projection_matrix, file_path_camera_transform)

        filename_camera_transform_world = f"cam_{image_name}_cam_proj_transform_world.pt"
        file_path_camera_transform = os.path.join(dataset_dir, filename_camera_transform_world)
        torch.save(camera.world_view_transform, file_path_camera_transform)

        filename_R = f"cam_{image_name}_R.pt"
        file_path_R = os.path.join(dataset_dir, filename_R)
        torch.save(cam_R, file_path_R)

        filename_T = f"cam_{image_name}_T.pt"
        file_path_T = os.path.join(dataset_dir, filename_T)
        torch.save(cam_T, file_path_T)

        lod_image_pair = {
            'camera_position': cam_position,
            'camera_dir': cam_dir,
            'camera_R': file_path_R,
            'camera_T': file_path_T,
            'lod_n_path': file_path_lod_n,
            'lod_x_path': file_path_lod_x,
            'lod_n': level_n,
            'lod_x': level_x,
            'image_name': str(image_name),
            'pixels_per_deg': pixels_per_deg,
            'levels': levels,
            'cam_proj_transform_path': file_path_camera_transform,
            'world_proj_transform_path': filename_camera_transform_world,
            'heatmap_path': file_path
        }

        dataset.add_entry(lod_image_pair)