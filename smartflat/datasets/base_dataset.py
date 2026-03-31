""""Smartflat abstract dataset class."""
import datetime
import logging
import os
import socket
import subprocess
import sys
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch

# Utils
from IPython.display import display


from smartflat.annotation_smartflat import AnnotationSmartflat
from smartflat.configs import (
    BaseSmartflatConfig,  # TODO refactor the constants to be configs related
)
from smartflat.constants import (
    available_modality,
    available_output_type,
    available_tasks,
    flag_columns,
    progress_cols,
)
from smartflat.datasets.build import generate_video_metadata
from smartflat.datasets.filter import filter_outlier_video_names
from smartflat.datasets.utils import append_annotations
from smartflat.datasets.visualization import show_video_metadata
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_collate import get_collate_subset
from smartflat.utils.utils_io import get_data_root, get_host_name


class BaseDataset(torch.utils.data.Dataset):  # type: ignore
    """Smartflat abstract dataset class.

    BaseDataset builds on top of torch map-style dataset class.

    All Smartflat should inherit from this class and implement
    the abstract methods below.
    """

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            int: number of samples in the dataset.
        """
        raise NotImplementedError
    
    def __init__(self):
        """Initialize dataset for training or evaluation.

        This function performs all the steps required to consume the dataset,
        for example downloading the data, unpacking, and performing any static
        pre-processing operations.

        This function should be idempotent, and avoid re-downloading data if not
        necessary.

        Raises:
            RuntimeError: raise an exception if initialization fails or is incomplete.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """Return training sample at position `index`.

        Must return a tuple with 3 elements:
        - input: tensor or numpy array with model inputs
        - label: tensor or numpy array with training target if any, or None
        - id: unique identifier for the sample in the dataset
        """

        raise NotImplementedError

class SmartflatDatasetBase(BaseDataset):
    """Dataset class for handling a multi-modal dataset
    
    
    Attributes:
        - metadata [pd.DataFrame]: Metadaata dataframe consisting in a row per consistent data (e.g. a video)
        
    Methods:
        - extraction: extract all available representations and copied them in the modality folder (e.g. root_dir/cuisine/ACCCHA/GoPro1/)
    """

    def __init__(self, dataset_name=None, root_dir=None, task_names=None, modality=None, scenario=None, do_apply_fixation=True, use_cached=False, verbose=False):

        if root_dir is None:
            self.root_dir = get_data_root()
        else:
            self.root_dir = root_dir


        self.dataset_name = dataset_name
        self.use_cached = use_cached
        self.do_apply_fixation = do_apply_fixation
        self.verbose = verbose

        if dataset_name == 'base':

            machine_dataset_name = get_host_name()

        else:

            machine_dataset_name = dataset_name

        self.metadata_path = os.path.join(self.root_dir, 'dataframes', 'persistent_metadata', '{}_dataset_df.csv'.format(machine_dataset_name))

        self.task_names = available_tasks if task_names == 'all' else task_names
        self.modality = available_modality if modality == 'all' else modality
        self.scenario = scenario      
                      
        purple(f'---------- smartflat dataset - root_dir: {self.root_dir} ---------- ')
        if self.verbose:
            green(f"Generating dataset stored in {self.metadata_path}")
        self.metadata = self.load_metadata()

        # TODO: for big training etc, we should add as confg whether loading this dataframe.
        # self.global_metadata = pd.read_csv(os.path.join(self.root_dir, 'dataframes', 'persistent_metadata', 'metadata.csv'))

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            int: number of samples in the dataset.
        """
        return len(self.metadata)

    def load_metadata(self) -> pd.DataFrame:
        """Load in metadata and capply static pre-processing."""

        if self.use_cached and os.path.isfile(self.metadata_path):
            print(f"Loading dataframe from: {self.metadata_path}")
            metadata = pd.read_csv(self.metadata_path)
            
            # Refresh un-serialized attributes
            metadata['annotation'] = metadata.apply(lambda x: AnnotationSmartflat(task_name=x.task_name), axis=1)
            #metadata = append_annotations(metadata, verbose=False)
            
        else:

            metadata = generate_video_metadata(self.root_dir, task_names=self.task_names, modality_to_explore=self.modality, do_apply_fixation=self.do_apply_fixation, verbose=self.verbose)

        # Update local base dataset
        if self.scenario == 'all' and not self.use_cached:
            self.update(metadata)
            
        return metadata

    def show(self) -> None:
        """Basic class-level summary of the dataset."""
        raise NotImplementedError

    def define_cohort(self, metadata, scenario):
        """Define a subset of the dataset based on a scenario.
        """
        raise NotImplementedError

    def update(self, metadata) -> None:

        current_date = datetime.datetime.now().strftime("%Y%m%d")

        if os.path.isfile(self.metadata_path):
            backup_path = self.metadata_path.replace('.csv', f'_backup_{current_date}.csv')
            subprocess.run(['mv', self.metadata_path, backup_path])
        # try:
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        metadata.to_csv(self.metadata_path, index=False)
        if self.verbose:
            print(f"Saved dataframe to: {self.metadata_path}")
        # except:
        #     backup_metadata_path = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', '{}_dataset_df.csv'.format(socket.gethostname()))
        #     metadata.to_csv(backup_metadata_path, index=False)
        #     print("[ERROR] Saving the local metadata to: {}".format(self.metadata_path))
        #     print("[SOLUTION] Saving to: {}".format(backup_metadata_path))
        
        
    def parse_annotations(self, verbose=True):
        """Parse the annotations for all the subjects in the dataset, 
        based on the all-anotations file.
        
        #TODO: set back to the SmartFlatDataset only to reduce redundancy , and update VideoBlockDataset.
        """
        def robust_parse_annotation(x, annot_all):
            x.annotation.parse(annot_all=annot_all, participant_id=x.participant_id, modality=x.folder_modality, verbose=verbose)#verbose)   

        if verbose:
            s1 = self.metadata.drop_duplicates(['participant_id'])['annotation_software'].value_counts().to_frame()
            s2 = self.metadata.drop_duplicates(['participant_id'])['has_annotation'].value_counts().to_frame()

        # Load annotations
        annotation_path = os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all.csv')
        annot_all = pd.read_csv(annotation_path)
        print(f'Use annotation file: {annotation_path}.')
        
        self.metadata['annotation'] = self.metadata.task_name.apply(lambda x: AnnotationSmartflat(task_name=x))

        self.metadata.apply(lambda x: robust_parse_annotation(x, annot_all), axis=1)
        

        # Reset attributes related to the annotations
        # self.metadata['has_annotation'] = self.metadata.apply(lambda x: 0 if  pd.isna(x.annotation) else
        #                                                                0 if (x.annotation.has_annotation == False) else
        #                                                                0 if (x.annotation.df is None) else
        #                                                                     1 if (x.annotation.has_annotation and len(x.annotation.df) > 0) else
        #                                                                     0, axis=1)
        
        self.metadata['has_annotation'] = self.metadata.apply(lambda x: x.annotation.has_annotation, axis=1)
        self.metadata['annotation_software'] = self.metadata.apply(lambda x: x.annotation.annotation_software if (x.has_annotation == 1) else
                                                                                        np.nan, axis=1)

        # self.metadata['annotation'] = self.metadata.groupby('participant_id')['annotation'].transform(
        #     lambda x: x.iloc[0] if x.notna().all() else np.nan
        # )
        
        if verbose:
            blue('State of annotations before parsing (based on per-subject files)')
            display(s1)
            display(s2)
            blue('State of annotations after parsing (based on per-subject files)')

        # Informative booleans
        display(self.metadata.groupby(['task_name', 'modality', 'has_annotation', 'annotation_software']).size().to_frame().rename(columns={0: 'N'}).T)
        
    def print_statistics(self) -> None:
        """Print dataset statistics."""
        raise NotImplementedError

    def log_summary_statistics(self):
            # Setup logging configuration
            log_dir = os.path.join(get_data_root(), 'log');  os.makedirs(log_dir, exist_ok=True)
            logging.basicConfig(filename=os.path.join(log_dir, 'dataset_summary_statistics.log'),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%Y-%m-%d %H:%M',
                                level=logging.INFO)
            print('Logging to', os.path.join(log_dir, 'dataset_summary_statistics.log'))

            logger = logging.getLogger()

            logger.info('---------------------------- Dataset summary statistics -----------------------------')
            logger.info(f'Number of unique subjects (trigram): {self.metadata.trigram.nunique()}')
            logger.info(f'Number of unique administrations (participant_id): {self.metadata.participant_id.nunique()}')
            logger.info(f'Number of unique modalities (N_m): {self.metadata.identifier.nunique()}')
            
            # Uncomment and log additional statistics if needed
            # logger.info(f'Average is_partition: {self.metadata.is_partition.mean():.2f}')
            # logger.info(f'Average is_consolidated: {self.metadata.is_consolidated.mean():.2f}')
            logger.info(f'Average processed: {self.metadata.processed.mean():.2f}')

            for output_type in available_output_type:
                n_succeded = (self.metadata[f'flag_{output_type}'] == 'success').sum()
                logger.info(f'N={n_succeded} {output_type}')  # Available and successful computation 
            
            # Log the metadata information
            logger.info('\n' + str(self.metadata['has_video'].value_counts().to_frame()))
            
            # Log specific flags count information
            flags_df = self.metadata[['flag_speech_recognition', 'flag_speech_representation',
                                    'flag_video_representation', 'flag_hand_landmarks',
                                    'flag_skeleton_landmarks', 'flag_collate_video']].apply(pd.Series.value_counts).fillna(0).astype(int)
            logger.info('\n' + str(flags_df))

class SmartflatDataset(SmartflatDatasetBase):
    """Dataset class for handling a multi-modal dataset
    
    
    Attributes:
        - metadata [pd.DataFrame]: Metadaata dataframe consisting in a row per consistent data (e.g. a video)
        
    Methods:
        - extraction: extract all available represenxtations and copied them in the odality folder (e.g. root_dir/cuisine/ACCCHA/GoPro1/)
    """

    def __init__(self, dataset_name='base', root_dir=None, scenario=None,  task_names=available_tasks, modality=available_modality, do_apply_fixation=True, verbose=False, use_cached=False):
        
        if root_dir is None:
            root_dir = get_data_root()

        super().__init__(dataset_name=dataset_name, root_dir=root_dir, task_names=task_names, modality=modality, scenario=scenario, do_apply_fixation=do_apply_fixation, verbose=verbose,  use_cached=use_cached)
        
        # ---- Report ---
        self.print_statistics()
        #self.log_summary_statistics()

        
    def show(self):
        
        for column in ['flag_video_representation', 'flag_hand_landmarks', 'flag_skeleton_landmarks']:
            display(self.metadata.groupby(['task_name', 'modality'])[column].value_counts().to_frame())

        display(self.metadata.groupby(['task_name', 'participant_id', 'modality'])['video_name'].apply(list).to_frame())
        
    def load_metadata(self) -> pd.DataFrame:
        """Load in metadata and apply static pre-processing."""
        
        # Creation of the metadata superset
        metadata = super().load_metadata()

        # Define cohort
        metadata = self.define_cohort(metadata, self.scenario)
        
        
        # Quality control checks
        # TODO: assert that the number of representation per modality is just one (no extra files loaded)
        if not len(metadata) == 0 and not (len(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)]) == 0):
            print(f"[WARNING] Number of instances per modality is higher than 1. Should collate or filter them.")
            #display(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)].sort_values(['task_name', 'participant_id', 'modality']))

        return metadata
    
    def define_cohort(self, metadata, scenario):
        """Define a subset of the dataset based on a scenario.
        """
        
        # Filters the videos that have been processed
        if scenario == 'all' or scenario is None:
            pass    
        
        elif scenario == 'collate':

            # Check if the video names are standard 
            metadata = filter_outlier_video_names(metadata)  
            metadata = get_collate_subset(metadata)
            
    
        elif scenario == 'present':
            
            metadata = metadata[(~metadata['video_path'].isna())].copy()
            
        elif scenario == 'unprocessed':
            
            pass#metadata = metadata[(~metadata['video_path'].isna())].copy()
        
            metadata = metadata[(~metadata['video_path'].isna()) & 
                                ((metadata['flag_speech_recognition'] == 'unprocessed') |
                                 (metadata['flag_speech_representation'] == 'unprocessed') |
                                 (metadata['flag_video_representation'] == 'unprocessed') |
                                 (metadata['flag_skeleton_landmarks'] == 'unprocessed') |
                                 (metadata['flag_hand_landmarks'] == 'unprocessed') | 
                                 (metadata['flag_collate_video'] == 'unprocessed')) 
                                
                                #((metadata['video_name'] == 'merged_video') | (metadata['n_videos'] == 1))
                                
                                ].copy()    
        
        
    
        else:
            raise ValueError(f"Scenario {scenario} is not registered. Use one of {['all', 'collate', 'present']}")
                
        return metadata
        
    def get_subject(self, identifier=None, participant_id = None, modality = None):
        """
        Get the metadata for a given subject providing the identifier or the participant_id
        Modality is optional
        """
        
        if identifier is not None:
            df = self.metadata[self.metadata['identifier'] == identifier]
        
        elif participant_id is not None:
            
            if modality is not None:
                
                df = self.metadata[(self.metadata['participant_id'] == participant_id) & (self.metadata['modality'] == modality)]
        
            else:
                df = self.metadata[self.metadata['participant_id'] == participant_id]
        
        return df
    
    def parse_annotations(self, verbose=True):
        """Parse the annotations for all the subjects in the dataset, 
        based on the all-anotations file.
        """
        def robust_parse_annotation(x, annot_all):
            x.annotation.parse(annot_all=annot_all, participant_id=x.participant_id, modality=x.folder_modality, verbose=verbose)#verbose)   

        if verbose:
            s1 = self.metadata.drop_duplicates(['participant_id'])['annotation_software'].value_counts().to_frame()
            s2 = self.metadata.drop_duplicates(['participant_id'])['has_annotation'].value_counts().to_frame()

        # Load annotations
        annotation_path = os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all.csv')
        annot_all = pd.read_csv(annotation_path)
        print(f'Use annotation file: {annotation_path}.')
        
        self.metadata['annotation'] = self.metadata.task_name.apply(lambda x: AnnotationSmartflat(task_name=x))

        self.metadata.apply(lambda x: robust_parse_annotation(x, annot_all), axis=1)
        

        # Reset attributes related to the annotations
        # self.metadata['has_annotation'] = self.metadata.apply(lambda x: 0 if  pd.isna(x.annotation) else
        #                                                                0 if (x.annotation.has_annotation == False) else
        #                                                                0 if (x.annotation.df is None) else
        #                                                                     1 if (x.annotation.has_annotation and len(x.annotation.df) > 0) else
        #                                                                     0, axis=1)
        
        self.metadata['has_annotation'] = self.metadata.apply(lambda x: x.annotation.has_annotation, axis=1)
        self.metadata['annotation_software'] = self.metadata.apply(lambda x: x.annotation.annotation_software if (x.has_annotation == 1) else
                                                                                        np.nan, axis=1)

        # self.metadata['annotation'] = self.metadata.groupby('participant_id')['annotation'].transform(
        #     lambda x: x.iloc[0] if x.notna().all() else np.nan
        # )
        
        if verbose:
            blue('State of annotations before parsing (based on per-subject files)')
            display(s1)
            display(s2)
            blue('State of annotations after parsing (based on per-subject files)')

        # Informative booleans
        display(self.metadata.groupby(['task_name', 'modality', 'has_annotation', 'annotation_software']).size().to_frame().rename(columns={0: 'N'}).T)
 
    def print_statistics(self):
        
        print('---------------------------- Dataset summary statistics -----------------------------')
        green('N.  unique subjects (trigram): {}, participant_id: {}, modalities: {}, processed: {:.2f}'.format(self.metadata.trigram.nunique(), self.metadata.participant_id.nunique(), self.metadata.identifier.nunique(), self.metadata.processed.mean()))

        expected_num_modalities = 4*self.metadata.participant_id.nunique()
        p_success = self.metadata.identifier.nunique() / expected_num_modalities
        green(f'[INFO] Number of expected modalities (4*N)={expected_num_modalities} ({p_success}) present)')
        yellow(f'[INFO] Average processed: {self.metadata.processed.mean():.2f}')

        # Stats
        # stats = show_video_metadata(self.metadata) #TODO: improve to include other statistcs
        # display(stats)

        # Informative booleans
        #display(self.metadata[['has_video', 'has_light_video']].apply(pd.Series.value_counts))#().to_frame())
        #display(self.metadata[flag_columns].apply(pd.Series.value_counts).fillna(0).astype(int))
        display(
            self.metadata.groupby(['modality'])[flag_columns]
            .apply(lambda df: df.apply(pd.Series.value_counts))
            .fillna(0)
            .astype(int)
        )
        
        if self.verbose:
            for output_type in available_output_type:
                n_succeded = (self.metadata[f'flag_{output_type}'] == 'success').sum()
                green(f'N={n_succeded} {output_type}') # Available and successful computation 
            display(self.metadata.drop_duplicates(['trigram']).groupby(['task_name'])['group'].value_counts().to_frame())
        # display(self.metadata[['flag_speech_recognition', 'flag_speech_representation','flag_video_representation',
        #                    'flag_hand_landmarks','flag_skeleton_landmarks', 'flag_collate_video'] ].apply(pd.Series.value_counts).fillna(0).astype(int))
        print('\n-----------------------------------------------------------------')

    def __len__(self) -> int:
        """Return dataset length.

        Returns:
            int: number of samples in the dataset.
        """
        return len(self.metadata)
