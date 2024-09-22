from lod_pairs_fvvdp import LODPairsFvvdp
from lod_fvvdp_continous import LODFvvdpEccentricityContinous
import random
import numpy as np
import torch
from tqdm import tqdm
from time import sleep

def sample_eccentricity():
    max_eccentricity_degrees = 27
    return random.uniform(0, max_eccentricity_degrees)

def sample_eccentricity_coordinates(img_width, img_height, eccentricity_deg, theta, pixels_per_deg):
    """
    Calculate the pixel coordinates for a given eccentricity degree value.
    Eccentricity is sampled in a random direction from the center of the image.
    """
    # Convert eccentricity in degrees to pixels
    eccentricity_pixels = eccentricity_deg * pixels_per_deg    

    # Calculate x, y coordinates
    x = int(img_width / 2 + eccentricity_pixels * np.cos(theta))
    y = int(img_height / 2 + eccentricity_pixels * np.sin(theta))
    
    # Ensure the coordinates are within image boundaries
    x = max(min(x, img_width - 1), 0)
    y = max(min(y, img_height - 1), 0)

    return x, y

def extract_patch(img, x, y, patch_size=64):
    """
    Extract a square patch of given size centered at (x, y).
    """
    img_height, img_width, _ = img.shape

    half_patch_size = patch_size // 2
    top = max(y - half_patch_size, 0)
    bottom = min(y + half_patch_size, img_height)
    left = max(x - half_patch_size, 0)
    right = min(x + half_patch_size, img_width)

    return img[top:bottom, left:right, :]

def extract_heatmap_patch(map, x, y, patch_size=64):
    """
    Extract a square patch of given size centered at (x, y).
    """
    img_height, img_width, _ = map.shape

    half_patch_size = patch_size // 2
    top = max(y - half_patch_size, 0)
    bottom = min(y + half_patch_size, img_height)
    left = max(x - half_patch_size, 0)
    right = min(x + half_patch_size, img_width)

    # Check if the computed patch is completely within the bounds of the image
    if (top == 0 and y - half_patch_size < 0) or (bottom == img_height and y + half_patch_size > img_height) or \
       (left == 0 and x - half_patch_size < 0) or (right == img_width and x + half_patch_size > img_width):
        return None

    return map[top:bottom, left:right]

def parse_tensor_string(tensor_str):
    # This function parses the numerical part of the tensor string
    numbers = tensor_str.strip("tensor([").rstrip("], dtype=torch.float64)")
    return np.fromstring(numbers, sep=',')

def get_ray_direction(x, y, image_width, image_height, projection_matrix, world_view_transform, device):
    x_ndc = (2.0 * x / image_width - 1.0) * (image_width / image_height)
    y_ndc = 1.0 - 2.0 * y / image_height

    # Point in NDC space
    point_ndc = torch.tensor([x_ndc, y_ndc, 1.0, 1.0], device=device)

    inv_proj_matrix = torch.linalg.inv(projection_matrix)
    point_camera = torch.matmul(inv_proj_matrix, point_ndc)

    direction_camera = torch.tensor([point_camera[0], point_camera[1], -1, 0], device=device)

    inv_view_matrix = torch.linalg.inv(world_view_transform)
    ray_direction_world = torch.matmul(inv_view_matrix, direction_camera)[:3]  # Discard the homogeneous coordinate

    # Normalize the direction vector
    ray_direction_world = ray_direction_world / torch.norm(ray_direction_world)

    return ray_direction_world

def pixel_to_ray(x, y, image_width, image_height, ppd, R):
    """
    Calculate the ray direction from the camera to a specified pixel, incorporating the camera's orientation.

    Args:
    x (int): x-coordinate of the pixel
    y (int): y-coordinate of the pixel
    image_width (int): Width of the image in pixels
    image_height (int): Height of the image in pixels
    ppd (float): Pixels per degree, how many pixels correspond to one degree of field of view
    R (numpy.ndarray): Rotation matrix of the camera in world coordinates

    Returns:
    torch.Tensor: The normalized direction vector (ray) from the camera to the pixel in world coordinates
    """
    # Calculate the center of the image
    center_x = image_width / 2
    center_y = image_height / 2

    # Calculate the angle in degrees that the pixel subtends from the optical axis
    angle_x = (x - center_x) / ppd
    angle_y = (y - center_y) / ppd

    # Convert angles to radians
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)

    # Calculate the direction vector components in camera coordinates
    z = np.cos(angle_x_rad) * np.cos(angle_y_rad)
    x = np.sin(angle_x_rad) * np.cos(angle_y_rad)
    y = np.sin(angle_y_rad)

    # Create the direction vector in camera coordinates
    direction_vector = np.array([x, y, z])

    # Normalize the direction vector
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

    # Transform the direction vector to world coordinates using the rotation matrix
    direction_vector_world = np.dot(R, direction_vector_normalized)

    return torch.tensor(direction_vector_world)


def linear_interpolation(discrete_values, difference_map, query_value):
    """
    Performs linear interpolation on a set of discrete values mapped to a difference map.
    
    Parameters:
    - discrete_values (list): The discrete values in the domain, e.g., [3, 2, 1, 0].
    - difference_map (list): The corresponding values in the difference map, e.g., [0.0, 0.0, 0.23650223, 1.6188014].
    - query_value (float): The value in the difference map to interpolate in the discrete domain.
    
    Returns:
    - float: The interpolated value in the discrete domain.
    """
    # Find the interval for interpolation
    for i in range(len(difference_map) - 1):
        if difference_map[i] <= query_value <= difference_map[i + 1] or difference_map[i] >= query_value >= difference_map[i + 1]:
            # Linear interpolation formula
            return discrete_values[i] + (discrete_values[i + 1] - discrete_values[i]) * (query_value - difference_map[i]) / (difference_map[i + 1] - difference_map[i])

    raise ValueError("Query value is out of the bounds of the difference map")



dataset_dir = "example_lod_pairs_white_5_levels"
lod_pairs_dataset = LODPairsFvvdp(dataset_dir)

ecc_dataset_dir_continous = "example_lod_ecc_white_5_levels_all"
import os
os.makedirs(ecc_dataset_dir_continous, exist_ok=True)
lod_ecc_dataset_cont= LODFvvdpEccentricityContinous(ecc_dataset_dir_continous, ecc_dataset_dir_continous)

views = lod_pairs_dataset.get_unique_view_indices()
print(views)
image_dim = lod_pairs_dataset.get_img_dimensions()
print(image_dim)
PPD = 32
JOD_THRESHOLD_DIFF = 1

for view_idx in tqdm(views):
    for _ in range(10):
        ecc = sample_eccentricity()
        # for each ecc sample a direction/theta
        for _ in range(10):
            theta = random.uniform(0, 2 * np.pi)
            x, y = sample_eccentricity_coordinates(image_dim[1], image_dim[0], ecc, theta, PPD)

            # start from lowest LOD
            levels = lod_pairs_dataset.get_num_lod_levels()
            entries = []

            for lod_x in list(range(levels)):
                pair = lod_pairs_dataset.get_by_image_name_and_lod_x(view_idx, lod_x)
                # get the lod average in this patch 

                ray_dir = pixel_to_ray(x, y, image_dim[1], image_dim[0], PPD, pair['camera_R'])

                lod_n_patch = extract_patch(pair['lod_n_image'], x, y)
                lod_x_patch = extract_patch(pair['lod_x_image'], x, y)

                heatmap = pair['heatmap'][0, :, 0, :, :].permute([1, 2, 0]).to(torch.float32).numpy()
                heatmap_patch = extract_heatmap_patch(heatmap, x, y)

                while heatmap_patch is None:
                    ecc = sample_eccentricity()
                    theta = random.uniform(0, 2 * np.pi)
                    x, y = sample_eccentricity_coordinates(image_dim[1], image_dim[0], ecc, theta, PPD)
                    heatmap_patch = extract_heatmap_patch(heatmap, x, y)

                heatmap_JOD_mean = np.mean(heatmap_patch)


                entry = {
                    'camera_position': pair['camera_position'],
                    'camera_dir': pair['camera_dir'],
                    'camera_R': pair['camera_R'],
                    'camera_T': pair['camera_T'],
                    'eccentricity': ecc,
                    'ray_dir': ray_dir,
                    'theta': theta,
                    'lod_n_patch': lod_n_patch,
                    'lod_x_patch': lod_x_patch,
                    'lod_n': pair['lod_n']/10,
                    'lod_x': pair['lod_x']/10,
                    'image_name': pair['image_name'],
                    'levels': pair['levels'],
                    'heatmap_patch': heatmap_patch,
                    'JOD_average': heatmap_JOD_mean
                }
                entries.append(entry)
            lod_ecc_dataset_cont.add_entries(entries)
    
