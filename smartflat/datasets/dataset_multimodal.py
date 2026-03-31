import os
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
from IPython.display import display


    
from smartflat.constants import available_modality, available_tasks
from smartflat.datasets.base_dataset import BaseDataset, SmartflatDatasetBase
from smartflat.datasets.build import generate_video_metadata
from smartflat.utils.utils_io import get_data_root


class MultimodalDataset(SmartflatDatasetBase):
    """Dataset class for handling a multi-modal dataset
    
    
    Attributes:
        - metadata [pd.DataFrame]: Metadaata dataframe consisting in a row per consistent data (e.g. a video)
        
    Methods:
        - extraction: extract all available representations and copied them in the odality folder (e.g. root_dir/cuisine/ACCCHA/GoPro1/)
    """

    def __init__(self, dataset_name='multimodal_dataset', root_dir=None, scenario=None,  task_names=available_tasks, modality=available_modality):
        
        
        super().__init__(dataset_name=dataset_name, root_dir=root_dir, modality=modality, scenario=scenario)
        
        
        # ---- Report ----
        print(f"Dataset: {self.__class__.__name__}")
        print(f"Modality: {modality}")
        print(f"Scenario: {scenario}")

        print("[Info] Subjects: {} | Admin: {}| Modality: {}/{} ".format(
            self.metadata.trigram.nunique(),
            self.metadata.participant_id.nunique(),
            len(self),
            self.global_metadata.drop_duplicates(['participant_id', 'modality']).identifier.nunique()
        ))
        
        
        
        
        
        if root_dir is None:
            self.root_dir = get_data_root()
        else:
            self.root_dir = root_dir
            
        self.task_names = task_names
        self.modality = ['Tobii', 'GoPro1', 'GoPro2', 'GoPro3'] if modality == 'all' else modality
        self.parse_video = parse_video
            
        self.metadata = self.load_metadata()
        self.global_metadata = pd.read_csv(os.path.join(self.root_dir,'dataframes', 'persistent_metadata', 'metadata.csv'))        
        
        
    def load_metadata(self) -> pd.DataFrame:
        """Load in metadata and apply static pre-processing."""
        
        return generate_video_metadata(self.root_dir, task_names=self.task_names, modality_to_explore=self.modality, parse_video=self.parse_video)
        
    
    def save_df(self):
        
        path = os.path.join(get_data_root(), 'dataframes/{}_dataset_df.csv'.format(socket.gethostname()))
        
        if os.path.isfile(path):
            subprocess.run(['mv', path, path.replace('.csv', '_backup.csv')])
        self.metadata.to_csv(path, index=False)
    
    
    def show(self):
        
        print("Show the modality having all embeddings of all videos computed")
        result = self.metadata.groupby(['task', 'participant_id', 'modality'])[['video_representation_computed', 'modality']].agg({'video_representation_computed': np.mean, 
                                                                                                                            'modality': 'count'}).rename(columns={'video_representation_computed': 'compute_perc',# 'Proportion of videos of that embedding to be computed'
                                                                                                                                                                        'modality': 'num_video_segments'}) # 'Number of videos in the folder'
        # Apply the styling to the DataFrame
        styled_result = result.head().style.applymap(color_embedding, subset=('compute_perc'))
        
        # Display dataframe
        display(styled_result)
        
        return 


# Define a function to apply color based on the values
def color_embedding(value):
    color = 'background-color: green' if value == 1 else 'background-color: red'
    return color
    
    
