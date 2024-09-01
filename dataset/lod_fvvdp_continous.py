import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class LODFvvdpEccentricityContinous(Dataset):
    def __init__(self, dataset_dir, root_dir):
        """
        Initialize the dataset by loading the metadata and setting the dataset directory.
        """
        self.dataset_dir = dataset_dir
        metadata_path = os.path.join(dataset_dir, 'metadata.csv')
        
        # Check if the metadata file exists, if not, create an empty DataFrame
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            # Define an empty DataFrame with the required columns
            self.metadata = pd.DataFrame(columns=['camera_position', 'eccentricity', 'theta', 'ray_dir',
                                                  'lod_n_path', 'lod_x_path', 'lod_n', 'lod_x', 'view_index',
                                                  'levels', 'heatmap_path', 'JOD_average'])
        
    def __len__(self):
        """
        Return the total number of patches in the dataset.
        """
        return len(self.metadata)
    
    def __getitem__(self, index):
        """
        Get an item by index.
        """
        meta = self.metadata.iloc[index]

        heatmap_patch = torch.load(meta['heatmap_path'])
        lod_n_patch = torch.load(meta['lod_n_path'])
        lod_x_patch = torch.load(meta['lod_x_path'])

        camera_R = torch.load(meta['camera_R'])
        camera_T = torch.load(meta['camera_T'])
    
        # Create the sample dictionary
        sample = {
            'camera_position': meta['camera_position'],
            'camera_dir': meta['camera_dir'],  
            'camera_R': camera_R,
            'camera_T': camera_T,
            'eccentricity': meta['eccentricity'],
            'theta': meta['theta'],
            'lod_n_patch': lod_n_patch,
            'lod_x_patch': lod_x_patch,
            'lod_n': meta['lod_n'],
            'lod_x': meta['lod_x'],
            'ray_dir': meta['ray_dir'],
            'view_index': meta['view_index'],
            'levels': meta['levels'],
            'heatmap': heatmap_patch,
            'JOD_average': meta['JOD_average']
        }

        return sample

    def add_entry(self, entry_data):
        """
        Save new data entry into the dataset.
        
        entry_data should include:
        - camera_position, eccentricity, ray_dir, lod_n_patch, lod_x_patch,
          lod_n, lod_x, view_index, pixels_per_deg, levels, heatmap_patch
        """
        # Generate unique filenames for the new patches
        base_filename = f"v{entry_data['view_index']}_e{entry_data['eccentricity']}_t{entry_data['theta']}"
        heatmap_filename = os.path.join(self.dataset_dir, f"{base_filename}_heatmap.pt")
        lod_n_filename = os.path.join(self.dataset_dir, f"{base_filename}_lod_n.pt")
        lod_x_filename = os.path.join(self.dataset_dir, f"{base_filename}_lod_x.pt")
        camera_R_filename = os.path.join(self.dataset_dir, f"{base_filename}_camera_R.pt")
        camera_T_filename = os.path.join(self.dataset_dir, f"{base_filename}_camera_T.pt")


        # Save the torch tensors
        torch.save(entry_data['lod_n_patch'], lod_n_filename)
        torch.save(entry_data['lod_x_patch'], lod_x_filename)
        torch.save(entry_data['heatmap_patch'], heatmap_filename)
        torch.save(entry_data['camera_R'], camera_R_filename)
        torch.save(entry_data['camera_T'], camera_T_filename)

        # Update the metadata
        new_entry = {
            'camera_position': entry_data['camera_position'],
            'camera_dir': entry_data['camera_dir'],  
            'camera_R': camera_R_filename,
            'camera_T': camera_T_filename,
            'eccentricity': entry_data['eccentricity'],
            'ray_dir': entry_data['ray_dir'],
            'theta': entry_data['theta'],
            'lod_n_path': lod_n_filename,
            'lod_x_path': lod_x_filename,
            'lod_n': entry_data['lod_n'],
            'lod_x': entry_data['lod_x'],
            'view_index': entry_data['view_index'],
            'levels': entry_data['levels'],
            'heatmap_path': heatmap_filename,
            'JOD_average': entry_data['JOD_average']
        }
        new_entry_df = pd.DataFrame([new_entry])  
        self.metadata = pd.concat([self.metadata, new_entry_df], ignore_index=True)

        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.metadata.to_csv(metadata_path, index=False)

    def add_entries(self, entries):
        """
        Save multiple data entries into the dataset.

        entries should be a list of dictionaries, each containing:
        - camera_position, eccentricity, ray_dir, lod_n_patch, lod_x_patch,
        lod_n, lod_x, view_index, pixels_per_deg, levels, heatmap_patch
        """
        new_entries = []
        
        for entry_data in entries:
            base_filename = f"v{entry_data['view_index']}_e{entry_data['eccentricity']}_t{entry_data['theta']}"
            heatmap_filename = os.path.join(self.dataset_dir, f"{base_filename}_heatmap.pt")
            lod_n_filename = os.path.join(self.dataset_dir, f"{base_filename}_lod_n.pt")
            lod_x_filename = os.path.join(self.dataset_dir, f"{base_filename}_lod_x.pt")
            camera_R_filename = os.path.join(self.dataset_dir, f"{base_filename}_camera_R.pt")
            camera_T_filename = os.path.join(self.dataset_dir, f"{base_filename}_camera_T.pt")

            torch.save(entry_data['lod_n_patch'], lod_n_filename)
            torch.save(entry_data['lod_x_patch'], lod_x_filename)
            torch.save(entry_data['heatmap_patch'], heatmap_filename)
            torch.save(entry_data['camera_R'], camera_R_filename)
            torch.save(entry_data['camera_T'], camera_T_filename)

            new_entry = {
                'camera_position': entry_data['camera_position'],
                'camera_dir': entry_data['camera_dir'],
                'camera_R': camera_R_filename,
                'camera_T': camera_T_filename,
                'eccentricity': entry_data['eccentricity'],
                'ray_dir': entry_data['ray_dir'],
                'theta': entry_data['theta'],
                'lod_n_path': lod_n_filename,
                'lod_x_path': lod_x_filename,
                'lod_n': entry_data['lod_n'],
                'lod_x': entry_data['lod_x'],
                'view_index': entry_data['view_index'],
                'levels': entry_data['levels'],
                'heatmap_path': heatmap_filename,
                'JOD_average': entry_data['JOD_average']
            }
            new_entries.append(new_entry)

        new_entries_df = pd.DataFrame(new_entries)
        self.metadata = pd.concat([self.metadata, new_entries_df], ignore_index=True)

        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.metadata.to_csv(metadata_path, index=False)

    def get_all_by_view_index(self, view_index):
        """
        Get all dataset items that have the specified view index.
        """
        filtered_samples = []
        for i in range(len(self.metadata)):
            if self.metadata.iloc[i]['view_index'] == view_index:
                filtered_samples.append(self.__getitem__(i))
        return filtered_samples
    
    def add_camera_parameter(self, cam_pos, cam_dir, camera_R, camera_T):
        """
        Add a single camera direction, rotation matrix, and translation vector to the dataset based on the given camera position.
        The rotation matrix and translation vector are saved as tensors in the specified directory.

        Args:
        cam_pos (tuple or str): The camera position, which will be used as a key.
        cam_dir (list): The camera direction vector.
        camera_R (torch.Tensor): The rotation matrix tensor.
        camera_T (torch.Tensor): The translation vector tensor.

        Returns:
        None. Updates the dataset's metadata in place.
        """
        # Convert camera position to a string if it's not already, to use as a key
        cam_pos_key = str(cam_pos)
    
        # Generate filenames for saving the tensors
        base_filename = f"camera_params_{cam_pos_key.replace(',', '_').replace(' ', '').replace('(', '').replace(')', '')}"
        camera_R_filename = os.path.join(self.dataset_dir, f"{base_filename}_R.pt")
        camera_T_filename = os.path.join(self.dataset_dir, f"{base_filename}_T.pt")

        # Save the tensors
        torch.save(camera_R, camera_R_filename)
        torch.save(camera_T, camera_T_filename)

        # Check if there are entries with the specified camera position
        if self.metadata['camera_position'].eq(cam_pos_key).any():
            # Update all entries with the same camera position
            self.metadata.loc[self.metadata['camera_position'] == cam_pos_key, 'camera_dir'] = cam_dir
            self.metadata.loc[self.metadata['camera_position'] == cam_pos_key, 'camera_R_path'] = camera_R_filename
            self.metadata.loc[self.metadata['camera_position'] == cam_pos_key, 'camera_T_path'] = camera_T_filename

        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.metadata.to_csv(metadata_path, index=False)