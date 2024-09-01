import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from lod_fvvdp_continous import LODFvvdpEccentricityContinous

def parse_tensor_string(tensor_str):
    numbers = tensor_str.strip("tensor([").rstrip("], dtype=torch.float64)")
    numbers = tensor_str.strip("tensor([").rstrip("], device='cuda:0')")
    return np.fromstring(numbers, sep=',')

def vector_to_angles(v):
    v_normalized = -v / np.linalg.norm(v)  
    azimuth = np.arctan2(v_normalized[1], v_normalized[0]) * (180 / np.pi)
    elevation = np.arcsin(v_normalized[2]) * (180 / np.pi)
    return elevation, azimuth


def update_plot(val):
    ax.clear()
    index = int(slider.val)
    samples = dataset.get_all_by_view_index(index)

    first = True

    for sample in samples:
        camera_position = parse_tensor_string(sample['camera_position'])
        ray_direction = parse_tensor_string(sample['ray_dir']) * 10
        camera_dir = parse_tensor_string(sample['camera_dir']) * 10

        if np.linalg.norm(camera_position) != 0:
            camera_position_normalized = camera_position / np.linalg.norm(camera_position)
        else:
            camera_position_normalized = camera_position

        ax.quiver(*camera_position_normalized, *ray_direction, length=0.1, color='blue')
        ax.quiver(*camera_position_normalized, *camera_dir, length=0.1, color='red')

        if first:
            elev, azim = vector_to_angles(camera_dir)

            ax.view_init(elev=-73.1364250495636, azim=-91.46171303604017) 

            print ("FIRST")
            first = False

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(sample['camera_position'])
    plt.draw()

metadata_path = 'playroom_lod_ecc_cont'
dataset = LODFvvdpEccentricityContinous(metadata_path, 'playroom_lod_ecc_cont')

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Index', 0, 14, valinit=0, valstep=1)
slider.on_changed(update_plot)

# Initialize
update_plot(0)

plt.show()