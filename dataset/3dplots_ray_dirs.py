import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from lod_fvvdp_continous import LODFvvdpEccentricityContinous
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def parse_tensor_string(tensor_str):
    numbers = tensor_str.strip("tensor([").rstrip("], dtype=torch.float64)")
    numbers = tensor_str.strip("tensor([").rstrip("], device='cuda:0')")
    return np.fromstring(numbers, sep=',')

def vector_to_angles(v):
    v_normalized = -v / np.linalg.norm(v) 
    azimuth = np.arctan2(v_normalized[1], v_normalized[0]) * (180 / np.pi)
    elevation = np.arcsin(v_normalized[2]) * (180 / np.pi)
    return elevation, azimuth

def update_patch_plot(ax_patch, patch_img):
    ax_patch.clear()
    imagebox = OffsetImage(patch_img, zoom=3)
    ab = AnnotationBbox(imagebox, (0.5, 0.5), frameon=False, box_alignment=(0.5, 0.5))
    ax_patch.add_artist(ab)
    ax_patch.axis('off')

def update_plot(val):
    ax.clear()
    ax_patch.clear()
    index = int(slider.val)
    samples = dataset.get_all_by_view_index(0)  # Single camera position; change 0 to desired index
    print(len(samples))
    if index < len(samples):
        sample = samples[index]
        camera_position = parse_tensor_string(sample['camera_position'])
        ray_direction = parse_tensor_string(sample['ray_dir']) * 10
        camera_dir = parse_tensor_string(sample['camera_dir']) * 10
        patch_img = sample['lod_n_patch']  

        # Normalize camera position
        if np.linalg.norm(camera_position) != 0:
            camera_position_normalized = camera_position / np.linalg.norm(camera_position)
        else:
            camera_position_normalized = camera_position

        ax.quiver(*camera_position_normalized, *ray_direction, length=0.1, color='blue')
        ax.quiver(*camera_position_normalized, *camera_dir, length=0.1, color='red')

        elev, azim = vector_to_angles(camera_dir)
        ax.view_init(elev=-73.1364250495636, azim=-91.46171303604017)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(f"Camera Position: {camera_position}")

        # Update patch subplot
        update_patch_plot(ax_patch, patch_img)

    plt.draw()

def on_key_press(event):
    if event.key == 'right':
        slider.set_val(min(slider.val + 1, slider.valmax))
    elif event.key == 'left':
        slider.set_val(max(slider.val - 1, slider.valmin))

metadata_path = 'playroom_lod_ecc_cont'
dataset = LODFvvdpEccentricityContinous(metadata_path, 'playroom_lod_ecc_cont')

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(121, projection='3d')
ax_patch = fig.add_subplot(122)
plt.subplots_adjust(bottom=0.25)

# Slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
samples = dataset.get_all_by_view_index(0)
slider = Slider(ax_slider, 'Index', 0, len(samples)-1, valinit=0, valstep=1)
slider.on_changed(update_plot)

# Initialize
update_plot(0)
fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()