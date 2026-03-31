"""Smartflat Skeleton landmarks Datasets."""
import os
import sys

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split


from smartflat.datasets.base_dataset import SmartflatDatasetBase
from smartflat.utils.utils_dataset import train_test_val_split_by
from smartflat.utils.utils_io import get_data_root
from smartflat.utils.utils_transform import zscore


class HandsDataset(SmartflatDatasetBase):
    """Dataset class for the hands landmarks estimation."""

    def __init__(
        self,
        dataset_name = 'hand_landmarks',
        root_dir: str = None,
        task_names: list = ['cuisine', 'lego'],
        modality: str = 'all',
        scenario: str = None,
        subject_filtering: str = None,
    ):
        self.modality = modality
        self.scenario=scenario
        self.subject_filtering = subject_filtering
        
        
        super().__init__(dataset_name=dataset_name, root_dir=root_dir, task_names=task_names, modality=modality, scenario=scenario)
        
        display(self.metadata.groupby(['task_name', 'modality'])['flag'].value_counts().to_frame())

        
        
    def load_metadata(self) -> pd.DataFrame:
        """Load in metadata and apply static pre-processing."""
        
        # Creation of the metadata superset
        metadata = super().load_metadata()
        
        # Normalize flag column for further processing
        metadata['flag'] = metadata.apply(lambda x: 'success' if x.hand_landmarks_computed else x.flag_hand_landmarks, axis=1)
        
        # Build train/val/test splits by participant_id .
        #metadata = train_test_val_split_by(metadata)
                
        # Define cohort
        metadata = self.define_cohort(metadata, self.scenario)
        
        
        # Quality control checks
        if not len(metadata) == 0 and not (len(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)]) == 0):
            print(f"[WARNING] Number of instances per modality is higher than 1. Should collate or filter them.")
            display(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)].sort_values(['task_name', 'participant_id', 'modality']))

        # Clean the columns
        #metadata = metadata[['task_name', 'trigram', 'participant_id', 'modality', 'identifier', 'video_name', 'date', 'annotation', 'video_representation_path', 'flag', 'fps', 'n_frames',  'duration', 'groupe', 'pathologie', 'ISDC', 'MoCA', 'video_path']]

        return metadata

    def __getitem__(self, index):
        meta_row = self.metadata.iloc[index]

        path = meta_row.hand_landmarks_path
        labels = meta_row.groupe
        ids = meta_row.identifier

        inputs = self.transform(path)
        labels = self.target_transform(labels)
        return inputs, labels, ids
    

    def transform(self, path) -> np.ndarray:
        """Transform the input data for the task."""
        raise NotImplementedError
    
    def target_transform(self, labels):
        """Transform the target data for the task.
        Identity function is the default.
        """
        return labels
    
    def task_target_transform(self, metadata) -> pd.Series:
        """Transform the `target_transform` outcome variable accounting for dataset-level statistics.
        Identity function is the default.
        """
        raise NotImplementedError
    
    def set_split(self, split):
        """Set the dataset to the appropriate training split.

        Note: this is likely to be deprecated.

        Args:
            split (str): Desired training split, one of ("train", "test", "val")
        """
        if split not in ("train", "test", "val"):
            raise ValueError

        self.metadata = self.metadata[self.metadata["split"].eq(split)].copy()
        
    def define_cohort(self, metadata, scenario):
        """Define a subset of the dataset based on a scenario.
        """
        
        n = len(metadata)
        # Filters the videos that have been processed
        if scenario == 'all' or scenario is None:
            pass    
        
        if 'unprocessed' in scenario:
            
            metadata = metadata[(~metadata['video_path'].isna()) & 
                                (metadata['flag'] == 'unprocessed') &  
                                ((metadata['video_name'] == 'merged_video') | (metadata['n_videos'] == 1))].copy()    
        

        if 'complete' in scenario:
            
            metadata = metadata[(metadata['flag'] == 'success') &  
                                ((metadata['video_name'] == 'merged_video') | (metadata['n_videos'] == 1))].copy()
            
            
        print(f"scenario {scenario}:  {n} -> {len(metadata)}")

        return metadata

class HandsProcessingDataset(SmartflatDatasetBase):
    """Dataset class for the hands landmarks estimation."""

    def __init__(
        self,
        dataset_name = 'tracking_hand_landmarks',
        root_dir: str = None,
        task_names: list = ['cuisine', 'lego'],
        modality: str = 'all',
        scenario: str = None,
        subject_filtering: str = None,
    ):
        self.modality = modality
        self.scenario=scenario
        self.subject_filtering = subject_filtering
        
        
        super().__init__(dataset_name=dataset_name, root_dir=root_dir, task_names=task_names, modality=modality, scenario=scenario)
        
        display(self.metadata.groupby(['task_name', 'modality'])['flag'].value_counts().to_frame())
        
    def load_metadata(self) -> pd.DataFrame:
        """Load in metadata and apply static pre-processing."""
        
        # Creation of the metadata superset
        metadata = super().load_metadata()
        
        # Normalize flag column for further processing
        metadata['flag'] = metadata.apply(lambda x: 'success' if x.tracking_hand_landmarks_computed else x.flag_tracking_hand_landmarks, axis=1)
        
        # Build train/val/test splits by participant_id .
        #metadata = train_test_val_split_by(metadata)
                
        # Define cohort
        metadata = self.define_cohort(metadata, self.scenario)
        
        
        # Quality control checks
        if not len(metadata) == 0 and not (len(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)]) == 0):
            print(f"[WARNING] Number of instances per modality is higher than 1. Should collate or filter them.")
            display(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)].sort_values(['task_name', 'participant_id', 'modality']))

        # Clean the columns
        #metadata = metadata[['task_name', 'trigram', 'participant_id', 'modality', 'identifier', 'video_name', 'date', 'annotation', 'video_representation_path', 'flag', 'fps', 'n_frames',  'duration', 'groupe', 'pathologie', 'ISDC', 'MoCA', 'video_path']]

        return metadata

    def __getitem__(self, index):
        meta_row = self.metadata.iloc[index]

        path = meta_row.hand_landmarks_path
        labels = meta_row.groupe
        ids = meta_row.identifier

        inputs = self.transform(path)
        labels = self.target_transform(labels)
        return inputs, labels, ids

    def transform(self, path) -> np.ndarray:
        """Transform the input data for the task."""
        raise NotImplementedError
    
    def target_transform(self, labels):
        """Transform the target data for the task.
        Identity function is the default.
        """
        return labels
    
    def task_target_transform(self, metadata) -> pd.Series:
        """Transform the `target_transform` outcome variable accounting for dataset-level statistics.
        Identity function is the default.
        """
        raise NotImplementedError
    
    def set_split(self, split):
        """Set the dataset to the appropriate training split.

        Note: this is likely to be deprecated.

        Args:
            split (str): Desired training split, one of ("train", "test", "val")
        """
        if split not in ("train", "test", "val"):
            raise ValueError

        self.metadata = self.metadata[self.metadata["split"].eq(split)].copy()
        
    def define_cohort(self, metadata, scenario):
        """Define a subset of the dataset based on a scenario.
        """
        
        n = len(metadata)
        # Filters the videos that have been processed
        if scenario == 'all' or scenario is None:
            pass    
        
        if 'unprocessed' in scenario:
            
            metadata = metadata[(~metadata['video_path'].isna()) & 
                                (metadata['flag'] == 'unprocessed') &  
                                ((metadata['video_name'] == 'merged_video') | (metadata['n_videos'] == 1))].copy()    
        

        if 'complete' in scenario:
            
            metadata = metadata[(metadata['flag'] == 'success') &  
                                ((metadata['video_name'] == 'merged_video') | (metadata['n_videos'] == 1))].copy()
            
            
        print(f"scenario {scenario}:  {n} -> {len(metadata)}")

        return metadata


if __name__ == "__main__":
    
    print(f"Benchmarking dataset: '{HandsDataset.__name__}'")
    dset = HandsDataset(root_dir=get_data_root(), scenario='all')
