import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class LODPairsFvvdp(Dataset):
    def __init__(self, dataset_dir):
        """
        Initialize the dataset by loading the metadata and setting the dataset directory.
        """
        self.dataset_dir = dataset_dir
        metadata_path = os.path.join(dataset_dir, 'metadata.csv')
        if not os.path.exists(metadata_path):
            self.metadata = pd.DataFrame(columns=['camera_position', 'camera_dir', 'camera_R', 'camera_T', 'lod_n_path', 'lod_x_path', 'lod_n', 'lod_x', 
                                                  'view_index', 'cam_proj_transform_path', 'world_proj_transform_path' ,'pixels_per_deg', 
                                                  'levels', 'heatmap_path'])
            self.metadata.to_csv(metadata_path, index=False)
        else:
            self.metadata = pd.read_csv(metadata_path)
        
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

        heatmap = torch.load(meta['heatmap_path'])
        lod_n_image = torch.load(meta['lod_n_path'])
        lod_x_image = torch.load(meta['lod_x_path'])
        cam_proj_transform = torch.load(meta['cam_proj_transform_path'])

        world_proj_path = os.path.join(self.dataset_dir, meta['world_proj_transform_path'])
        world_proj_transform = torch.load(world_proj_path)
        
        camera_R = torch.load(meta['camera_R'])
    
        camera_T = torch.load(meta['camera_T'])

        # Create the sample dictionary
        sample = {
            'camera_position': meta['camera_position'],
            'camera_dir': meta['camera_dir'],
            'camera_R': camera_R,
            'camera_T': camera_T,
            'lod_n_image': lod_n_image,
            'lod_x_image': lod_x_image,
            'lod_n': meta['lod_n'],
            'lod_x': meta['lod_x'],
            'view_index': meta['view_index'],
            'cam_proj_transform': cam_proj_transform,
            'world_proj_transform': world_proj_transform,
            'pixels_per_deg': meta['pixels_per_deg'],
            'levels': meta['levels'],
            'heatmap': heatmap
        }

        return sample
    

    def get_all_by_view_index(self, view_index):
        """
        Get all dataset items that have the specified view index.
        """
        filtered_samples = []
        for i in range(len(self.metadata)):
            if self.metadata.iloc[i]['view_index'] == view_index:
                filtered_samples.append(self.__getitem__(i))
        return filtered_samples

    def get_unique_view_indices(self):
        """
        Return a list of unique view indices from the dataset.
        """
        return self.metadata['view_index'].unique().tolist()
    
    def get_img_dimensions(self):

        img_dim = self.__getitem__(0)['lod_n_image'].shape

        return img_dim
    
    def get_by_view_index_and_lod_x(self, view_index, lod_x):
        """
        Get all dataset items that have the specified view index.
        """
        for i in range(len(self.metadata)):
            if self.metadata.iloc[i]['view_index'] == view_index and self.metadata.iloc[i]['lod_x'] == lod_x:
                return self.__getitem__(i)

    def get_num_lod_levels(self):
        """
        Return a list of unique view indices from the dataset.
        """
        return self.metadata['lod_x'].unique().tolist()[-1] + 1
    
    def add_entry(self, new_metadata_entry):
        """
        Add a new entry to the dataset and update the metadata file.

        Parameters:
        - new_data: A dictionary containing all necessary data for the new entry.
                    Expected keys: 'camera_position', 'lod_n_image', 'lod_x_image', 'lod_n', 'lod_x',
                                    'view_index', 'cam_proj_transform', 'pixels_per_deg', 'levels', 'heatmap'
        """
        # Append the new entry to the metadata DataFrame
        new_entry_df = pd.DataFrame([new_metadata_entry])
        self.metadata = pd.concat([self.metadata, new_entry_df], ignore_index=True)

        # Save the updated metadata to CSV
        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.metadata.to_csv(metadata_path, index=False)