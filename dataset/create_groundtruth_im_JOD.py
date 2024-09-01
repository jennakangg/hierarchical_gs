from lod_pairs_fvvdp import LODPairsFvvdp
from lod_fvvdp_eccentricity import LODFvvdpEccentricity
from lod_fvvdp_continous import LODFvvdpEccentricityContinous
import numpy as np
import torch
from tqdm import tqdm


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

    return map[top:bottom, left:right]

def parse_tensor_string(tensor_str):
    # This function parses the numerical part of the tensor string
    numbers = tensor_str.strip("tensor([").rstrip("], dtype=torch.float64)")
    return np.fromstring(numbers, sep=',')


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



dataset_dir = "playroom_lod_pairs"
lod_pairs_dataset = LODPairsFvvdp(dataset_dir)

views = lod_pairs_dataset.get_unique_view_indices()
image_dim = lod_pairs_dataset.get_img_dimensions()

PPD = 32
JOD_THRESHOLD_DIFF = 1
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
def save_combined_image(original_image, actual_image, heatmap, filename='combined_output.png'):
    """
    Saves a combined image showing the original image and the LOD output image side by side.
    
    Parameters:
    - original_image (numpy.ndarray): The original image data.
    - actual_image (numpy.ndarray): The LOD output image data.
    - filename (str): Filename to save the combined image.
    """
    nan_mask = np.isnan(original_image)
    unique_numbers = np.unique(original_image[~nan_mask])  
    
    colors = plt.cm.get_cmap('tab20', len(unique_numbers) + 1)  
    color_map = {}
    patches = []  

    for i, num in enumerate(unique_numbers):
        color_map[num] = colors(i)
        patches.append(mpatches.Patch(color=colors(i), label=str(num)))
    

    color_map["nan"] = (0, 0, 0, 1)  # RGBA for black
    patches.append(mpatches.Patch(color=(0, 0, 0, 1), label='NaN'))

    colored_original_image = np.zeros((original_image.shape[0],original_image.shape[1], 4)) 
    for num in np.append(unique_numbers, np.nan):  # Include NaN in the loop
        if np.isnan(num) or num == "nan":
            print(num)
            colored_original_image[nan_mask] = (0, 0, 0, 1)
        else:
            colored_original_image[original_image==num] = color_map[num]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  
    

    im = axes[0].imshow(colored_original_image)
    
    plt.legend(handles=patches, bbox_to_anchor=(0, 1.02, 1, .102), loc='lower left', mode="expand", borderaxespad=0., ncol=1)

    axes[0].set_title('Colored LOD Output')
    axes[0].axis('off')

    # Plot LOD output image
    axes[1].imshow(actual_image, interpolation='nearest')  
    axes[1].set_title('Actual Image')
    axes[1].axis('off')

    # Plot heatmap
    axes[2].imshow(heatmap, interpolation='nearest')
    axes[2].set_title('Heatmap')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

patch_sizes = 32
output_image = np.full((image_dim[0], image_dim[1]), np.nan)

for view_idx in tqdm(views[0:3]):
    for y in range(0, image_dim[0], patch_sizes):  
        for x in range(0, image_dim[1], patch_sizes):

            if x + patch_sizes > image_dim[1] or y + patch_sizes > image_dim[0]:  
                continue  

            levels = lod_pairs_dataset.get_num_lod_levels()
            difference_map = []

            for lod_x in range(levels):
                pair = lod_pairs_dataset.get_by_view_index_and_lod_x(view_idx, lod_x)
                
                heatmap = pair['heatmap'][0, :, 0, :, :].permute([1, 2, 0]).to(torch.float32).numpy()
                heatmap_patch = extract_heatmap_patch(heatmap, x, y, patch_sizes)  

                heatmap_JOD_mean = np.mean(heatmap_patch)

                if np.linalg.norm(heatmap_patch) == 0:
                    heatmap_JOD_mean = 0

                # difference_map.append(heatmap_JOD_mean)
                
                # # Check if JOD mean exceeds the threshold
                # if heatmap_JOD_mean >= JOD_THRESHOLD_DIFF:
                #     discrete_values = list(reversed(range(lod_x, levels)))
                #     continuous_LOD_val = linear_interpolation(discrete_values, difference_map, JOD_THRESHOLD_DIFF)
                    
                #     # Save the continuous LOD value in the corresponding position in the output image
                #     output_image[y:y+patch_sizes, x:x+patch_sizes] = continuous_LOD_val                    
                #     # print(continuous_LOD_val)
                #     lod_n_image = pair['lod_n_image']
                #     break
                
                if heatmap_JOD_mean <= JOD_THRESHOLD_DIFF:                    
                    output_image[y:y+patch_sizes, x:x+patch_sizes] = lod_x                    
                    # print(continuous_LOD_val)
                    lod_n_image = pair['lod_n_image']
                    break
    
    save_combined_image(output_image, np.array(pair['lod_x_image']), heatmap, f'lod_output_{view_idx}_32.png')
 
