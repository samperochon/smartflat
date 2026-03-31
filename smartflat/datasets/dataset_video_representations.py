"""Smartflat Videos Block Representations Datasets."""

import os
import sys
import json
from os.path import join as j_

import numpy as np
import pandas as pd
from tqdm import tqdm

from IPython.display import display
from scipy.cluster.vq import whiten as scipy_whiten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist


from smartflat.configs import BaseSmartflatConfig
from smartflat.configs.change_points_config import ChangePointDetectionDeploymentConfig
from smartflat.configs.clustering_config import ClusteringSegmentsConfig
from smartflat.configs.loader import import_config, load_config
from smartflat.constants import flag_columns, progress_cols
from smartflat.datasets.base_dataset import SmartflatDatasetBase
from smartflat.datasets.transform import whiten_matrix
from smartflat.tests.test_dataset import test_dataset_video_representations
from smartflat.utils.utils import add_cols_suffixes, pairwise, smartflat_range
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_dataset import train_test_val_split_by
from smartflat.utils.utils_io import get_data_root, load_pca
from smartflat.utils.utils_transform import zscore

# leftovers TODO model from dataset_tv/dataet
# from hcdatasets.transforms.augmentations import TimeseriesAugmentations


class VideoBlockDataset(SmartflatDatasetBase):
    """Dataset class for video block representations"""

    def __init__(
        self,
        root_dir: str = None,
        annotator_id: str = "samperochon",
        round_number: int = 0,
        dataset_name="video_block_representation",
        task_names: list = ["cuisine"],
        modality: str = "all",
        scenario: str = None,
        normalization: str = "identity",
        whiten: bool = True, 
        n_pca_components: int = None,
        # crop_edges_config_name: str = None,    
        verbose=False
    ):
        self.normalization = normalization
        self.n_pca_components = n_pca_components
        self.whiten = whiten
        self.annotator_id = annotator_id
        self.round_number = round_number
        #self.crop_edges_config_name = crop_edges_config_name
        

        super().__init__(
            dataset_name=dataset_name,
            root_dir=root_dir,
            task_names=task_names,
            modality=modality,
            scenario=scenario,
            verbose=verbose
        )

        assert not (self.normalization == "PCA" and self.n_pca_components is None)

        self.load_pca()
        
        self.metadata.sort_values('task_number_int', inplace=True)
        
        # ---- Report ---
        self.print_statistics()
        #self.log_summary_statistics()
        
        # if self.crop_edges_config_name is not None:
        #     pass#assert 'test_bounds' in  self.metadata.columns



    def load_metadata(self) -> pd.DataFrame:
        """Load in metadata and apply static pre-processing."""

        
        metadata = super().load_metadata()
       
       
        # Creation of the metadata superset
        # if self.crop_edges_config_name is not None:
        
        #     from smartflat.features.symbolization.utils_dataset import (
        #         get_experiments_dataframe,
        #     )
        #     print(f'Loading metadata with edges estimates from {self.crop_edges_config_name} to access test_bounds.')

        #     # Creation of the metadata superset
        #     bdf = get_experiments_dataframe(experiment_config_name=self.crop_edges_config_name,  return_symbolization=True)

        #     #Merge with the metadata
        #     cols = ['participant_id', 'modality', 'opening_size', 'closing_size',  'test_bounds', 'test_bounds_timestamps', 'processed_N', 'edges_durations', 'left_bounds_embeddings', 'right_bounds_embeddings']
        #     metadata = metadata.merge(bdf[cols], on=['participant_id', 'modality'], how='left', indicator=False)
            
        #     #Print the number of element without tests_bounds
        #     n = len(metadata)
        #     metadata = metadata[metadata['test_bounds'].notna()]
        #     print(f"Discarding {n - len(metadata)} samples with missing test_bounds. -> N={len(metadata)}.")
        # else:
        #     red(f'/!\ Skipping edges removal.')
            
        # Normalize flag column for further processing
        metadata["flag"] = metadata.apply(
            lambda x: (
                "success"
                if (
                    x.video_representation_computed
                    and (x.flag_video_representation == "success")
                )
                else x.flag_video_representation
            ),
            axis=1,
        )

        # Build train/val/test splits by participant_id .

        metadata = train_test_val_split_by(metadata, train_subjects=[], train_size=0.8, test_to_val_size=1.0, split_by='participant_id')
        metadata['split'] = metadata['split'].apply(lambda x: x if x != 'val' else 'test')
        display(metadata.groupby('split').task_number_int.agg(np.sum).to_frame().rename(columns={'task_number_int': 'Sum participant ids'}))

        #metadata['split'] = 'train'
        # Define cohort
        metadata = self.define_cohort(metadata, self.scenario)
        

        # Quality control checks
        # TODO: assert that the number of representation per modality is just one (no extra files loaded)
        if not len(metadata) == 0 and not (len(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)]) == 0):
            print(f"[WARNING] Number of instances per modality is higher than 1. Should collate or filter them.")
            display(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)].sort_values(['task_name', 'participant_id', 'modality']))

            
        # Clean the columns
        #metadata = filter_progress_cols(metadata, progress_cols)

        return metadata

    def load_pca(self):

        # Load extra models if needed #TODO: remove multiple names... use only univariate boolean configs.
        if self.normalization in ["PCA", "PCA-Z-score", "PCA+l1", "PCA+l2"] and (
            self.n_pca_components is not None
        ):
            assert (len(self.task_names) == 1) and (len(self.modality) == 1)

            # try:
            self.pca = load_pca(
                self.task_names[0],
                self.modality[0],
                self.n_pca_components,
                whiten=self.whiten,
            )
            # except FileNotFoundError:
            #    print('PCA pre-trained eigenvectors not found. Skipping.'
            
    def __getitem__(self, index):
        meta_row = self.metadata.iloc[index]

        inputs = np.load(meta_row.video_representation_path)
    
        # Crop the video edges
        start, end = meta_row.test_bounds
        inputs = inputs[start:end, :]
        #print(f'Cropping video edges: {start} -> {end}.')
            
        labels = meta_row.group
        ids = meta_row.identifier

        inputs = self.transform(inputs)
        labels = self.target_transform(labels)
        return inputs, labels, ids
    
    
    
    def print_statistics(self):

        blue('---------------------------- Dataset summary statistics -----------------------------')
        green('Number of unique subjects (trigram): {}, participant_id: {}, modalities: {}'.format(self.metadata.trigram.nunique(), self.metadata.participant_id.nunique(), self.metadata.identifier.nunique()))

        expected_num_modalities = 4*self.metadata.participant_id.nunique()
        #p_success = self.metadata.identifier.nunique() / expected_num_modalities
        #green(f'[INFO] Number of expected modalities (4*N)={expected_num_modalities} ({p_success}) present)')


        # Stats
        # display(
        #     self.metadata.groupby(["task_name", "modality"])['flag']
        #     .value_counts()
        #     .to_frame().transpose()
        # )


        # Informative booleans
        #display(self.metadata[['has_video', 'has_light_video']].apply(pd.Series.value_counts))#().to_frame())
        #display(self.metadata.drop_duplicates(['trigram']).groupby(['task_name'])['diagnosis_group'].value_counts().to_frame())
        # display(self.metadata[['flag_speech_recognition', 'flag_speech_representation','flag_video_representation',
        #                    'flag_hand_landmarks','flag_skeleton_landmarks', 'flag_collate_video'] ].apply(pd.Series.value_counts).fillna(0).astype(int))

    def show(self):
        """TODO: make deprecated"""

        # Define a function to apply color based on the values
        def binary_color(value):
            color = "background-color: green" if value == 1 else "background-color: red"
            return color

        print("Show the modality having all embeddings of all videos computed")
        result = (
            self.metadata.groupby(["task", "participant_id", "modality"])[
                ["video_representation_computed", "modality"]
            ]
            .agg({"video_representation_computed": np.mean, "modality": "count"})
            .rename(
                columns={
                    "video_representation_computed": "compute_perc",  # 'Proportion of videos of that embedding to be computed'
                    "modality": "num_video_segments",
                }
            )
        )  # 'Number of videos in the folder'
        # Apply the styling to the DataFrame
        styled_result = result.head().style.applymap(
            binary_color, subset=("compute_perc")
        )

        # Display dataframe
        display(styled_result)

        return

    def transform(self, inputs) -> np.ndarray:
        """Transform the input data for the task."""
    
        eps = 1e-8

        if self.normalization == "identity":
            pass
        elif self.normalization == "Z-score":
            inputs = zscore(inputs)
        elif self.normalization == "l1":
            inputs = inputs / (
                np.linalg.norm(inputs, ord=1, axis=1, keepdims=True) + eps
            )
        elif self.normalization == "l2":
            inputs = inputs / (
                np.linalg.norm(inputs, ord=2, axis=1, keepdims=True) + eps
            )
        elif self.normalization == "PCA" and self.n_pca_components is not None:
            inputs = self.pca.transform(inputs)

        elif self.normalization == "PCA+l1" and self.n_pca_components is not None:
            inputs = self.pca.transform(inputs)
            inputs = inputs / (
                np.linalg.norm(inputs, ord=1, axis=1, keepdims=True) + eps
            )

        elif self.normalization == "PCA+l2" and self.n_pca_components is not None:
            inputs = self.pca.transform(inputs)
            inputs = inputs / (
                np.linalg.norm(inputs, ord=2, axis=1, keepdims=True) + eps
            )

        elif self.normalization == "whiten":
            inputs = scipy_whiten(inputs)
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        return inputs.astype(np.float32)

    def target_transform(self, labels):
        """Transform the target data for the task.
        Identity function is the default.
        """
        return labels

    def task_target_transform(self, metadata) -> pd.Series:
        """Transform the `target_transform` outcome variable accounting for dataset-level statistics.
        Identity function is the default.
        """
        return metadata.feature_column

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
        """Define a subset of the dataset based on a scenario."""

        n = len(metadata)
        # Filters the videos that have been processed
        if scenario == "all" or scenario is None:
            pass

        if "unprocessed" in scenario:

            metadata = metadata[
                (~metadata["video_path"].isna())
                & (metadata["flag"] == "unprocessed")
                & (
                    (metadata["video_name"] == "merged_video")
                    | (metadata["n_videos"] == 1)
                )
            ].copy()

        if "complete" in scenario:

            metadata = metadata[
                (metadata["flag"] == "success")
                & (
                    (metadata["video_name"] == "merged_video")
                    | (metadata["n_videos"] == 1)
                )
            ].copy()

        if "success" in scenario:

            #print('Displaying "flag" counts:')
            #display(metadata.flag.value_counts())

            # Filtering by "success" flag
            initial_count = len(metadata)
            metadata = metadata[metadata["flag"] == "success"].copy()
            print(f"[INFO] Filtered by 'success' flag: {initial_count} -> {len(metadata)}")

            # Filtering by video conditions
            #initial_count = len(metadata)
            #metadata = metadata[
            #    (metadata["video_name"] == "merged_video") | (metadata["n_videos"] == 1)
            #].copy()
            #print(f"[INFO] Filtered by video conditions ('merged_video' or 'n_videos' == 1): {initial_count} -> {len(metadata)}")

            # Checking uniqueness of identifiers per modality
            metadata["num_identifier_per_modality"] = metadata.groupby(
                ["task_name", "participant_id", "modality"]
            )["identifier"].transform(lambda x: len(np.unique(x)))

            if not (metadata["num_identifier_per_modality"] == 1).all():
                print(
                    "[WARNING] Not all rows have exactly one unique identifier per modality. "
                    "Check the following rows:"
                )
                problematic_rows = metadata[metadata["num_identifier_per_modality"] != 1]
                display(problematic_rows)
            else:
                print("[INFO] All rows have exactly one unique identifier per modality.")
                del metadata["num_identifier_per_modality"]

            
            # Remove rows with missing duration and fps (but add fps=25 for Tobii)
            n_na = metadata.fps.isna().sum()
            metadata.loc[(metadata["fps"].isna()) & (metadata['modality']=='Tobii'), "fps"] = 25
            
            print(f'Added fps for {n_na-metadata.fps.isna().sum()} missing fps values for Tobii modality.')
            
            n = len(metadata)
            metadata = metadata[~metadata[["fps"]].isna().any(axis=1)].copy()
            print(f"Removed {n-len(metadata)} rows with missing fps.")
            
        print(f"Extra filtering using scenario {scenario}: {n} -> {len(metadata)}")

        return metadata

    def load_embeddings(self):
        self.metadata["embeddings"] = [self[i][0] for i in range(len(self.metadata))]

    def load_change_point_detection(
        self,
        annotator_id,
        round_number,
        config_name="ChangePointDetectionDeploymentConfig",
    ):
        """TODO: clean the function"""

        config = import_config(config_name)
        change_point_experiment_ids=config.experiment_ids
        experiment_name=config.experiment_name
        self.metadata['cpts_config_name'] = config_name
        
        assert isinstance(change_point_experiment_ids, list)

        # Multiple change points scales
        if len(change_point_experiment_ids) == 2:  

            results_folder = os.path.join(self.root_dir, "experiments", experiment_name)

            if not (
                os.path.isdir(
                    os.path.join(results_folder, change_point_experiment_ids[0])
                )
                and os.path.isdir(
                    os.path.join(results_folder, change_point_experiment_ids[1])
                )
            ):
                raise FileNotFoundError(
                    f"{os.path.join(results_folder, change_point_experiment_ids[0])} not found."
                )

            for lambda_i, experiment_id in enumerate(change_point_experiment_ids):
                experiment_folder = os.path.join(results_folder, experiment_id, annotator_id, f'round_{round_number}')
                self.metadata[f"cpts_{lambda_i}"] = self.metadata.identifier.apply(
                    lambda x: (
                        np.load(os.path.join(experiment_folder, f"{x}_cpts.npy")).astype(int)
                        if os.path.isfile(
                            os.path.join(experiment_folder, f"{x}_cpts.npy")
                        )
                        else np.nan
                    )
                )
                self.metadata[f"n_cpts_{lambda_i}"] = self.metadata[
                    f"cpts_{lambda_i}"
                ].apply(lambda x: len(x) if type(x) in [list, np.ndarray] else np.nan)
                
            #print("[WARNING] Added cpts_0 and cpts_1 and n_cpts")
            
            
            n = len(self.metadata)
            #display(self.metadata[self.metadata['cpts_0'].isna()])
            self.metadata.dropna(subset=["cpts_0"], inplace=True)
            #red('Prevent removing missing cpts')
            print(
                f"Discarding {n - len(self.metadata)} samples with missing change-points. -> N={len(self.metadata)}."
            )

            # For now, by convention, we take the first lambda to be the one used for the experiment
            self.metadata["cpts"] = self.metadata["cpts_1"]
            self.metadata["n_cpts"] = self.metadata["n_cpts_1"]
            self.metadata["n_cpts_raw"] = self.metadata["n_cpts_1"]
            
            
        elif len(change_point_experiment_ids) == 1:

            change_point_experiment_ids = change_point_experiment_ids[0]

            results_folder = os.path.join(self.root_dir, "experiments", experiment_name, change_point_experiment_ids, annotator_id, f'round_{round_number}')

            if not os.path.isdir(results_folder):
                raise FileNotFoundError(
                    f"{results_folder} not found."
                )
            self.metadata[f"cpts"] = self.metadata.identifier.apply(
                lambda x: (
                    np.load(os.path.join(results_folder, f"{x}_cpts.npy")).astype(int)
                    if os.path.isfile(os.path.join(results_folder, f"{x}_cpts.npy"))
                    else np.nan
                )
            )
            self.metadata[f"n_cpts"] = self.metadata[f"cpts"].apply(
                lambda x: len(x) if type(x) in [list, np.ndarray] else np.nan
            )
            
            # Find missing cpts rows
            #display(self.metadata[self.metadata.identifier.apply(lambda x: os.path.exists(os.path.join(experiment_folder, f"{x}_cpts.npy")))])

            n = len(self.metadata)
            self.metadata.dropna(subset=["cpts"], inplace=True)
            print(
                f"Discarding {n - len(self.metadata)} samples with missing change-points. -> N={len(self.metadata)}."
            )
        else:
            raise NotImplementedError
        
        self.metadata['raw_cpts'] = self.metadata['cpts'].copy()
        self.metadata['n_cpts_raw'] = self.metadata['raw_cpts'].apply(lambda x: len(x) if type(x) == np.ndarray else np.nan).to_list()
        
        green(f"Loading temporal segmentation (two-scales) from {results_folder}")

    def load_clustering_labels(self, config_name="ClusteringDeploymentConfig", add_distances=False, load_clustering=False, annotator_id = None, round_number=None):


        config = import_config(config_name)
        
        if annotator_id is None:
            annotator_id = self.annotator_id
        if round_number is None:
            round_number = self.round_number
    
        config.dataset_params['annotator_id'] = annotator_id
        config.dataset_params['round_number'] = round_number

        #normalize_distance_per_cluster = config.normalize_distance_per_cluster
        if config.task_type == 'symbolization':
            clustering_config_name = config.clustering_config_name
            clustering_config = import_config(config.clustering_config_name)
        else:
            clustering_config_name = config_name
            clustering_config = config
            
        self.metadata['clustering_config_name'] = clustering_config_name
        self.metadata['kernel_name'] = config.model_params['kernel_name']
        
        normalization = clustering_config.model_params['normalization']
        
        self.clustering_config_name = clustering_config_name

        if load_clustering or config.task_type == "clustering":
            
            if round_number is None:
                experiment_folder = os.path.join(self.root_dir, "experiments", clustering_config.experiment_name, clustering_config.experiment_id)
            else:
                experiment_folder = os.path.join(self.root_dir, "experiments", clustering_config.experiment_name, clustering_config.experiment_id, annotator_id, f'round_{round_number}')
            
            if not os.path.isdir(experiment_folder):
                raise FileNotFoundError(f"{experiment_folder} not found.")
            
            self.metadata["embedding_labels"] = self.metadata.identifier.apply(
            lambda x: (
                np.load(os.path.join(experiment_folder,  f"{x}_labels.npy")).astype(int)
                if os.path.isfile(os.path.join(experiment_folder, f"{x}_labels.npy"))
                else np.nan)
            )
            self.metadata["embedding_labels_path"] = self.metadata.identifier.apply(
            lambda x: (
                os.path.join(experiment_folder,  f"{x}_labels.npy")
                if os.path.isfile(os.path.join(experiment_folder, f"{x}_labels.npy"))
                else np.nan)
            )
            display(self.metadata[['embedding_labels_path', "embedding_labels"]].head(3))

                            
        elif config.task_type == "symbolization":

            inference_method = config.model_params['inference_method']
            
            if inference_method in ['wasserstein_projection', 'subspace_projection']:
                
                experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', 'inference')                    
                #print(f"Loaded clustering labels from {config_name} (inference_method={inference_method}) from {experiment_folder}")
                if not os.path.isdir(experiment_folder):
                    raise FileNotFoundError(f"{experiment_folder} not found.")

                self.metadata["embedding_labels"] = self.metadata.apply(
                            lambda x: (np.load(os.path.join(experiment_folder, f'{config_name}_prototypes_projection_labels_{x.identifier}.npy')).astype(int)
                            if os.path.isfile(os.path.join(experiment_folder, f'{config_name}_prototypes_projection_labels_{x.identifier}.npy'))
                            else np.nan), axis=1)


            elif inference_method == 'clustering':
                
                clustering_config = import_config(config.clustering_config_name)
                    
                if round_number is None:
                    experiment_folder = os.path.join(self.root_dir, "experiments", clustering_config.experiment_name, clustering_config.experiment_id)
                else:
                    experiment_folder = os.path.join(self.root_dir, "experiments", clustering_config.experiment_name, clustering_config.experiment_id, annotator_id, f'round_{round_number}')
            
                
                print(f"Loaded clustering labels from {config_name} (inference_method={inference_method}) from {experiment_folder}")

                if not os.path.isdir(experiment_folder):
                    raise FileNotFoundError(f"{experiment_folder} not found.")
                                
                self.metadata["embedding_labels"] = self.metadata.apply(
                lambda x: (
                    np.load(os.path.join(experiment_folder, f'{x.identifier}_labels.npy')).astype(int)
                    if os.path.isfile(os.path.join(experiment_folder, f'{x.identifier}_labels.npy'))
                    else np.nan), axis=1)
                
            else:
                raise ValueError(f"Invalid inference_method: {inference_method}")
        
    
        # 2) Post-processing 
        # self.metadata["has_clustering_labels"] = self.metadata.embedding_labels.notna()
        # print('n has label=', self.metadata['has_clustering_labels'].sum() )
        # if self.metadata['has_clustering_labels'].sum() < len(self.metadata):
        #     red('Rows without clustering labels:')
        #     display(self.metadata[self.metadata['has_clustering_labels'] == False].head(2)[['identifier']])
        #     n = len(self.metadata)
        #     self.metadata = self.metadata[self.metadata['has_clustering_labels'] == True]
        #     print(f"Discarding {n - len(self.metadata)} samples with missing embedding_labels. -> N={len(self.metadata)}.")   
        
        #display(self.metadata.embedding_labels.head(5))
    
        self.metadata["raw_embedding_labels"] = self.metadata["embedding_labels"].copy()

        # Add paths to the distance matrix
        experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', 'inference')
        self.metadata['M_path'] = self.metadata.apply(lambda x: os.path.join(experiment_folder, f'{config_name}_distance_matrix_{x.identifier}.npy'), axis=1)
        
        #r_experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}','distance_matrices_reduction'); 
        #self.metadata['M_reduced_path'] = self.metadata.apply(lambda x: os.path.join(r_experiment_folder, f'{config_name}_reduced_distance_matrix_{x.identifier}.npy'), axis=1)
        # 3) Compute cluster assignement distance if needed
        if add_distances:
            from smartflat.features.symbolization.utils import compute_centroid_distances
            # for i, row in self.metadata.iterrows():
                # if os.path.isfile(row.M_path):
                #     #print(f'Loading distance matrix from {row.M_path}')
                #     #print(np.load(row.M_path).shape)
                # else:
                #     raise FileNotFoundError(f"Distance matrix file not found: {row.M_path}")
            #self.metadata = compute_centroid_distances(self.metadata, config_name=config_name, normalization=normalization, normalize_per_cluster=normalize_distance_per_cluster)
            self.metadata['cluster_dist'] =  self.metadata.apply(lambda x: np.min(np.load(x.M_path), axis=1), axis=1)
            print(f'Loaded distaces from e.g {self.metadata.M_path.iloc[0]}')
        green(f"Loaded clustering labels from {experiment_folder}")


    def load_compute_hyperparameters(self, verbose=False):

        hyperparams_table_path = os.path.join(
            self.root_dir, "dataframes", "persistent_metadata", "hyperparams_table.csv"
        )
        hyperparams_table = pd.read_csv(hyperparams_table_path)

        self.metadata = self.metadata.merge(
            hyperparams_table[
                ["identifier", "delta_t", "segment_length", "num_frames"]
            ],
            on=["identifier"],
            how="left",
            indicator=True,
        ).rename(columns={"_merge": "hyperparameter_metadata_merge"})

        if verbose:
            pass#display(self.metadata["hyperparameter_metadata_merge"].value_counts())

        # Fix: backfill unique constant value
        self.metadata["segment_length"].fillna(16, inplace=True)
        self.metadata["segment_length"] = self.metadata["segment_length"].astype(int)
        self.metadata["delta_t"].fillna(8, inplace=True)
        self.metadata["delta_t"] = self.metadata["delta_t"].astype(int)
        self.metadata["token_duration"] = (
            self.metadata["segment_length"] / self.metadata["fps"]
        )

        # # We exclude for now the ebedding array that don't seem to match in terms of shape of X and expected shape of X (based on indices)
        self.metadata["idx_embedding"] = self.metadata.apply(
            lambda x: (
                list(
                    smartflat_range(
                        num_frames=int(x.n_frames),
                        segment_length=int(x.segment_length),
                        delta_t=int(x.delta_t),
                    )
                )
                if not np.isnan(x.n_frames)
                else np.nan
            ),
            axis=1,
        )

        self.metadata["n_idx_embedding"] = self.metadata.idx_embedding.apply(
             lambda x: len(x) if not np.isnan(x).all() else np.nan
         )

        self.metadata["has_hyperparams"] = self.metadata.num_frames.apply(
            lambda x: False if np.isnan(x) else True
        )
    
    def load_gamma(self, config_name, annotator_id, round_number, temperature=1.0,  overwrite=False):

        config = import_config(config_name)
        gamma_path = os.path.join(get_data_root(), 'outputs', config.experiment_name, annotator_id, f'round_{round_number}', f'gamma_estimates_t{temperature}.json')
        os.makedirs(os.path.dirname(gamma_path), exist_ok=True)
        print(f'Created {os.path.dirname(gamma_path)}')
        if  gamma_path is not None and os.path.isfile(gamma_path) and not overwrite:
            with open(gamma_path, 'r') as f:
                gamma_est = json.load(f)
            
            self.metadata['gamma'] = self.metadata.identifier.map(gamma_est)
            return
        else:
            print(f'File not found: {gamma_path}. Estimating gamma values...')
        
        gamma_est = {}
        for i in tqdm(range(len(self))):

            X, _, ids = self[i]
            condensed_sq_dists = pdist(X, metric='sqeuclidean')
            sigma_rbf = np.median(np.sqrt(condensed_sq_dists))
            gamma = 1 / (2 * temperature * sigma_rbf ** 2)
            
            gamma_est[ids] = gamma

        self.metadata['gamma'] = self.metadata.identifier.map(gamma_est)
        
        with open(gamma_path, 'w') as f:
            json.dump(gamma_est, f)
        print(f'Saved gamma estimates in {gamma_path}')

    def load_optimal_lambdas(self, config_name, annotator_id, round_number):

        config = import_config(config_name)

        optimal_lambdas_path = os.path.join(
            self.root_dir, "experiments", config.experiment_name, annotator_id, f'round_{round_number}', "lambda_optimal.csv"
        )
        if not os.path.isfile(optimal_lambdas_path):
            raise ValueError(
                f"No results of estimated regularization parameters found: {optimal_lambdas_path}."
            )

        print("Loading penalty parameters in {}.".format(optimal_lambdas_path))

        if config.model_params["penalty"] == "modality-specific":
            usecols = [
                "identifier",
                "L_hat^c",
                "k_hat^c",
                "x0_hat^c",
                "L_var^c",
                "k_var^c",
                "x0_var^c",
                "lambda_0^c",
                "lambda_1^c",
            ]

            subjects_df = pd.read_csv(optimal_lambdas_path, usecols=usecols)
            subjects_df.rename(
                columns={
                    "L_hat^c": "L_hat",
                    "k_hat^c": "k_hat",
                    "x0_hat^c": "x0_hat",
                    "L_var^c": "L_var",
                    "k_var^c": "k_var",
                    "x0_var^c": "x0_var",
                    "lambda_0^c": "lambda_0",
                    "lambda_1^c": "lambda_1",
                },
                inplace=True,
            )

        elif config.model_params["penalty"] == "personalized":
            usecols = [
                "identifier",
                "L_hat^p",
                "k_hat^p",
                "x0_hat^p",
                "L_var^p",
                "k_var^p",
                "x0_var^p",
                "lambda_0^p",
                "lambda_1^p",
            ]

            subjects_df = pd.read_csv(optimal_lambdas_path, usecols=usecols)
            subjects_df.rename(
                columns={
                    "L_hat^p": "L_hat",
                    "k_hat^p": "k_hat",
                    "x0_hat^p": "x0_hat",
                    "L_var^p": "L_var",
                    "k_var^p": "k_var",
                    "x0_var^p": "x0_var",
                    "lambda_0^p": "lambda_0",
                    "lambda_1^p": "lambda_1",
                },
                inplace=True,
            )
        else:
            raise ValueError(f'Unknown penalty type: {config.model_params["penalty"]}')

        self.metadata = self.metadata.merge(subjects_df, on="identifier", how="left")
        
        
        print(
            "Retrieve optimal lambda_0 and lambda_1 for {}/{} identifiers ({} mode).".format(
                len(self.metadata.dropna(subset=["lambda_0"])),
                self.metadata.identifier.nunique(), 
                config.model_params["penalty"],
            )
        )

class PrototypesTrajectoryDataset(VideoBlockDataset):
    """Dataset class for prototypes subspace representations"""

    def __init__(
        self,
        root_dir: str = None,
        dataset_name="prototypes_trajectory_representation",
        config_name='SymbolicInferenceConfig',
        task_names: list = ["cuisine"],#, "lego"],
        annotator_id: str = "samperochon",
        round_number: int = 0,
        modality: str = "all",
        scenario: str = None,
        normalization: str = "l2",
        use_reduced_distance_matrix: bool = True,
        input_space: str = 'K_space',
        whiten: bool = True, 
        n_pca_components: int = None,
        #crop_edges_config_name: str = None,
        verbose=False):
        
        self.config_name = config_name
        #self.crop_edges_config_name = crop_edges_config_name
        
        self.normalization = normalization
        self.n_pca_components = n_pca_components
        self.whiten = whiten
        self.use_reduced_distance_matrix = use_reduced_distance_matrix
        self.input_space = input_space
        #self.crop_edges_config_name = crop_edges_config_name
        super().__init__(
            dataset_name=dataset_name,
            root_dir=root_dir,
            annotator_id=annotator_id,
            round_number=round_number,
            task_names=task_names,
            modality=modality,
            scenario=scenario,
            verbose=verbose
        )

        assert not (self.normalization == "PCA" and self.n_pca_components is None)

        self.load_pca()
        
        # ---- Report ---
        self.print_statistics()
        #self.log_summary_statistics()
        
        # if self.crop_edges_config_name is not None:
        #     assert 'test_bounds' in  self.metadata.columns


    def load_metadata(self) -> pd.DataFrame:
        """Load in metadata and apply static pre-processing."""

        #TODO: fix this 
        from smartflat.features.symbolization.utils_dataset import get_experiments_dataframe

        # Creation of the metadata superset
        #metadata = super().load_metadata()
        print('Annotator id: {}, round number: {}'.format(self.annotator_id, self.round_number))
        metadata = get_experiments_dataframe(experiment_config_name=self.config_name, annotator_id=self.annotator_id, round_number=self.round_number, return_symbolization=True)
        
        config = import_config(self.config_name)
        print(config, config.to_dict())

        experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id,  self.annotator_id, f'round_{self.round_number}', 'distance_matrices_reductions')
        metadata['M_path'] = metadata.apply(lambda x: os.path.join(experiment_folder, f'{self.config_name}_distance_matrix_{x.identifier}.npy'), axis=1)
        
        if self.input_space == 'K_space':
            metadata['M_reduced_path'] = metadata.apply(lambda x: os.path.join(experiment_folder, f'{self.config_name}_reduced_distance_matrix_{x.identifier}.npy'), axis=1)
        elif self.input_space == 'G_space':
            metadata['M_reduced_path']  = metadata.apply(lambda x: os.path.join(experiment_folder, f'{self.config_name}_reduced_G_space_distance_matrix_{x.identifier}.npy'), axis=1)

        # Normalize flag column for further processing
        metadata["flag"] = metadata.apply(
            lambda x: (
                "success"
                if (
                    x.video_representation_computed
                    and (x.flag_video_representation == "success")
                )
                else x.flag_video_representation
            ),axis=1,)

        
        # Build train/val/test splits by participant_id .
        # metadata = train_test_val_split_by(metadata)

        # Define cohort
        metadata = self.define_cohort(metadata, self.scenario)
        
        # Quality control checks
        # TODO: assert that the number of representation per modality is just one (no extra files loaded)
        if not len(metadata) == 0 and not (len(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)]) == 0):
            print(f"[WARNING] Number of instances per modality is higher than 1. Should collate or filter them.")
            display(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)].sort_values(['task_name', 'participant_id', 'modality']))

        # Clean the columns
        #metadata = filter_progress_cols(metadata, progress_cols)

        return metadata

    def __getitem__(self, index):
        meta_row = self.metadata.iloc[index]

        if self.use_reduced_distance_matrix:
            path = meta_row.M_reduced_path
        else:
            path = meta_row.M_path
            
        inputs = np.load(path)
        
        #start, end = meta_row.test_bounds
        #inputs = inputs[start:end, :]
        #print(f'Cropping video edges: {start} -> {end}.')
            
        labels = meta_row.group
        ids = meta_row.identifier

        inputs = self.transform(inputs)
        labels = self.target_transform(labels)
        return inputs, labels, ids


  
class VideoSymbolsDataset(SmartflatDatasetBase):
    """Dataset class for video symbolization, i.e. video block representation (see `VideoBlockDataset`) with added
    temporal segmentation (cpts) and clustering labels. One symbolic scale only is added.
    
    """

    def __init__(
        self,
        root_dir: str = None,
        dataset_name="video_block_representation_augmented",
        task_names: list = ["cuisine", "lego"],
        modality: str = "all",
        scenario: str = None,
        normalization: str = "whiten",
        whiten: bool = False,  # Active only for normalization=PCA
        n_pca_components: int = None,
        verbose=False
    ):
        self.normalization = normalization
        self.n_pca_components = n_pca_components
        self.whiten = whiten

        super().__init__(
            dataset_name=dataset_name,
            root_dir=root_dir,
            task_names=task_names,
            modality=modality,
            scenario=scenario,
            verbose=verbose
        )

        assert not (self.normalization == "PCA" and self.n_pca_components is None)
        
        # Load articat
        self.load_pca()
        

        # ---- Report ---
        self.print_statistics()
        #self.log_summary_statistics()


    def load_metadata(self) -> pd.DataFrame:
        """Load in metadata and apply static pre-processing."""

        # Creation of the metadata superset
        metadata = super().load_metadata()

        # Normalize flag column for further processing
        metadata["flag"] = metadata.apply(
            lambda x: (
                "success"
                if (
                    x.video_representation_computed
                    and (x.flag_video_representation == "success")
                )
                else x.flag_video_representation
            ),
            axis=1,
        )

        # Build train/val/test splits by participant_id .
        # metadata = train_test_val_split_by(metadata)

        # Define cohort
        metadata = self.define_cohort(metadata, self.scenario)
        

        # Quality control checks
        # TODO: assert that the number of representation per modality is just one (no extra files loaded)
        if not len(metadata) == 0 and not (len(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)]) == 0):
            print(f"[WARNING] Number of instances per modality is higher than 1. Should collate or filter them.")
            display(metadata[metadata.duplicated(['task_name', 'participant_id', 'modality'], keep=False)].sort_values(['task_name', 'participant_id', 'modality']))

            
        # Clean the columns
        metadata = filter_progress_cols(metadata, progress_cols)

        return metadata

    def load_pca(self):

        # Load extra models if needed #TODO: remove multiple names... use only univariate boolean configs.
        if self.normalization in ["PCA", "PCA-Z-score", "PCA+l1", "PCA+l2"] and (
            self.n_pca_components is not None
        ):
            assert (len(self.task_names) == 1) and (len(self.modality) == 1)

            # try:
            self.pca = load_pca(
                self.task_names[0],
                self.modality[0],
                self.n_pca_components,
                whiten=self.whiten,
            )
            # except FileNotFoundError:
            #    print('PCA pre-trained eigenvectors not found. Skipping.'
            
    def __getitem__(self, index):
        meta_row = self.metadata.iloc[index]

        path = meta_row.video_representation_path
        labels = meta_row.group
        ids = meta_row.identifier

        inputs = self.transform(path)
        labels = self.target_transform(labels)
        return inputs, labels, ids
    
    def print_statistics(self):

        blue('---------------------------- Dataset summary statistics -----------------------------')
        green('Number of unique subjects (trigram): {}, participant_id: {}, modalities: {}'.format(self.metadata.trigram.nunique(), self.metadata.participant_id.nunique(), self.metadata.identifier.nunique()))

        expected_num_modalities = 4*self.metadata.participant_id.nunique()
        p_success = self.metadata.identifier.nunique() / expected_num_modalities
        green(f'[INFO] Number of expected modalities (4*N)={expected_num_modalities} ({p_success}) present)')


        # Stats
        display(
            self.metadata.groupby(["task_name", "modality"])['flag']
            .value_counts()
            .to_frame().transpose()
        )


        # Informative booleans
        #display(self.metadata[['has_video', 'has_light_video']].apply(pd.Series.value_counts))#().to_frame())
        #display(self.metadata.drop_duplicates(['trigram']).groupby(['task_name'])['diagnosis_group'].value_counts().to_frame())
        # display(self.metadata[['flag_speech_recognition', 'flag_speech_representation','flag_video_representation',
        #                    'flag_hand_landmarks','flag_skeleton_landmarks', 'flag_collate_video'] ].apply(pd.Series.value_counts).fillna(0).astype(int))


    def show(self):
        """TODO: make deprecated"""

        # Define a function to apply color based on the values
        def binary_color(value):
            color = "background-color: green" if value == 1 else "background-color: red"
            return color

        print("Show the modality having all embeddings of all videos computed")
        result = (
            self.metadata.groupby(["task", "participant_id", "modality"])[
                ["video_representation_computed", "modality"]
            ]
            .agg({"video_representation_computed": np.mean, "modality": "count"})
            .rename(
                columns={
                    "video_representation_computed": "compute_perc",  # 'Proportion of videos of that embedding to be computed'
                    "modality": "num_video_segments",
                }
            )
        )  # 'Number of videos in the folder'
        # Apply the styling to the DataFrame
        styled_result = result.head().style.applymap(
            binary_color, subset=("compute_perc")
        )

        # Display dataframe
        display(styled_result)

        return

    def transform(self, path) -> np.ndarray:
        """Transform the input data for the task."""
        inputs = np.load(path)

        eps = 1e-8

        if self.normalization == "identity":
            pass
        elif self.normalization == "Z-score":
            inputs = zscore(inputs)
        elif self.normalization == "l1":
            inputs = inputs / (
                np.linalg.norm(inputs, ord=1, axis=1, keepdims=True) + eps
            )
        elif self.normalization == "l2":
            inputs = inputs / (
                np.linalg.norm(inputs, ord=2, axis=1, keepdims=True) + eps
            )
        elif self.normalization == "PCA" and self.n_pca_components is not None:
            inputs = self.pca.transform(inputs)

        elif self.normalization == "PCA+l1" and self.n_pca_components is not None:
            inputs = self.pca.transform(inputs)
            inputs = inputs / (
                np.linalg.norm(inputs, ord=1, axis=1, keepdims=True) + eps
            )

        elif self.normalization == "PCA+l2" and self.n_pca_components is not None:
            inputs = self.pca.transform(inputs)
            inputs = inputs / (
                np.linalg.norm(inputs, ord=2, axis=1, keepdims=True) + eps
            )

        elif self.normalization == "whiten":
            inputs = scipy_whiten(inputs)
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        return inputs.astype(np.float32)

    def target_transform(self, labels):
        """Transform the target data for the task.
        Identity function is the default.
        """
        return labels

    def task_target_transform(self, metadata) -> pd.Series:
        """Transform the `target_transform` outcome variable accounting for dataset-level statistics.
        Identity function is the default.
        """
        return metadata.feature_column

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
        """Define a subset of the dataset based on a scenario."""

        n = len(metadata)
        # Filters the videos that have been processed
        if scenario == "all" or scenario is None:
            pass

        if "unprocessed" in scenario:

            metadata = metadata[
                (~metadata["video_path"].isna())
                & (metadata["flag"] == "unprocessed")
                & (
                    (metadata["video_name"] == "merged_video")
                    | (metadata["n_videos"] == 1)
                )
            ].copy()

        if "complete" in scenario:

            metadata = metadata[
                (metadata["flag"] == "success")
                & (
                    (metadata["video_name"] == "merged_video")
                    | (metadata["n_videos"] == 1)
                )
            ].copy()

        if "success" in scenario:

            #print('Displaying "flag" counts:')
            #display(metadata.flag.value_counts())

            # Filtering by "success" flag
            initial_count = len(metadata)
            metadata = metadata[metadata["flag"] == "success"].copy()
            print(f"[INFO] Filtered by 'success' flag: {initial_count} -> {len(metadata)}")

            # Filtering by video conditions
            #initial_count = len(metadata)
            #metadata = metadata[
            #    (metadata["video_name"] == "merged_video") | (metadata["n_videos"] == 1)
            #].copy()
            #print(f"[INFO] Filtered by video conditions ('merged_video' or 'n_videos' == 1): {initial_count} -> {len(metadata)}")

            # Checking uniqueness of identifiers per modality
            metadata["num_identifier_per_modality"] = metadata.groupby(
                ["task_name", "participant_id", "modality"]
            )["identifier"].transform(lambda x: len(np.unique(x)))

            if not (metadata["num_identifier_per_modality"] == 1).all():
                print(
                    "[WARNING] Not all rows have exactly one unique identifier per modality. "
                    "Check the following rows:"
                )
                problematic_rows = metadata[metadata["num_identifier_per_modality"] != 1]
                display(problematic_rows)
            else:
                print("[INFO] All rows have exactly one unique identifier per modality.")
                del metadata["num_identifier_per_modality"]

        print(f"Extra filtering using scenario {scenario}: {n} -> {len(metadata)}")

        return metadata

    def load_embeddings(self):
        self.metadata["embeddings"] = [self[i][0] for i in range(len(self.metadata))]

    def load_change_point_detection(
        self,
        config_name="ChangePointDetectionDeploymentConfig",
    ):
        """TODO: clean the function"""

        config = import_config(config_name)
        change_point_experiment_ids=config.experiment_ids
        experiment_name=config.experiment_name
        
        assert isinstance(change_point_experiment_ids, list)

        # Multiple change points scales
        if len(change_point_experiment_ids) == 2:  # See if we scale

            results_folder = os.path.join(self.root_dir, "experiments", experiment_name)

            if not (
                os.path.isdir(
                    os.path.join(results_folder, change_point_experiment_ids[0])
                )
                and os.path.isdir(
                    os.path.join(results_folder, change_point_experiment_ids[1])
                )
            ):
                raise FileNotFoundError(
                    f"{os.path.join(results_folder, change_point_experiment_ids[0])} not found."
                )

            for lambda_i, experiment_id in enumerate(change_point_experiment_ids):
                experiment_folder = os.path.join(results_folder, experiment_id)
                self.metadata[f"cpts_{lambda_i}"] = self.metadata.identifier.apply(
                    lambda x: (
                        np.load(os.path.join(experiment_folder, f"{x}_cpts.npy")).astype(int)
                        if os.path.isfile(
                            os.path.join(experiment_folder, f"{x}_cpts.npy")
                        )
                        else np.nan
                    )
                )
                self.metadata[f"n_cpts_{lambda_i}"] = self.metadata[
                    f"cpts_{lambda_i}"
                ].apply(lambda x: len(x) if type(x) in [list, np.ndarray] else np.nan)
                self.metadata[f"K_{lambda_i}"] = (
                    self.metadata[f"n_cpts_{lambda_i}"] - 2
                )  # Remove starting '0' and trailing 'n_frames'

            #print("[WARNING] Added cpts_0 and cpts_1 and n_cpts")
            
            
            n = len(self.metadata)
            self.metadata.dropna(subset=["cpts_0"], inplace=True)
            #red('Prevent removing missing cpts')
            print(
                f"Discarding {n - len(self.metadata)} samples with missing change-points. -> N={len(self.metadata)}."
            )

            # For now, by convention, we take the first lambda to be the one used for the experiment
            self.metadata["cpts"] = self.metadata["cpts_0"]
            self.metadata["n_cpts"] = self.metadata["n_cpts_0"]
            self.metadata["K"] = (
                self.metadata["n_cpts"] - 2
            )  # Remove starting '0' and trailing 'n_frames'

        elif len(change_point_experiment_ids) == 1:

            change_point_experiment_ids = change_point_experiment_ids[0]

            results_folder = os.path.join(self.root_dir, "experiments", experiment_name)

            if not os.path.isdir(
                os.path.join(results_folder, change_point_experiment_ids)
            ):
                raise FileNotFoundError(
                    f"{os.path.join(results_folder, change_point_experiment_ids)} not found."
                )

            experiment_folder = os.path.join(
                results_folder, change_point_experiment_ids
            )
            self.metadata[f"cpts"] = self.metadata.identifier.apply(
                lambda x: (
                    np.load(os.path.join(experiment_folder, f"{x}_cpts.npy")).astype(int)
                    if os.path.isfile(os.path.join(experiment_folder, f"{x}_cpts.npy"))
                    else np.nan
                )
            )
            self.metadata[f"n_cpts"] = self.metadata[f"cpts"].apply(
                lambda x: len(x) if type(x) in [list, np.ndarray] else np.nan
            )

            n = len(self.metadata)
            self.metadata.dropna(subset=["cpts"], inplace=True)
            print(
                f"Discarding {n - len(self.metadata)} samples with missing change-points. -> N={len(self.metadata)}."
            )
        else:
            raise NotImplementedError

        
        green(f"Loading temporal segmentation (ntwo-scales) from {experiment_folder}")

    # def load_clustering_labels(self, config_name="ClusteringDeploymentConfig"):

    # Deprecated: re-adapt to the main one
    #     clustering_config = import_config(config_name)
    #     experiment_folder = os.path.join(
    #         self.root_dir,
    #         "experiments",
    #         clustering_config.experiment_name,
    #         clustering_config.experiment_id,
    #     )
    #     green(f"Loading clustering labels from {experiment_folder}")

    #     if not os.path.isdir(experiment_folder):
    #         raise FileNotFoundError(f"{experiment_folder} not found.")
        
    #     self.clustering_config = clustering_config
    #     self.metadata["embedding_labels"] = self.metadata.identifier.apply(
    #         lambda x: (
    #             np.load(os.path.join(experiment_folder, f"{x}_labels.npy")).astype(int)
    #             if os.path.isfile(os.path.join(experiment_folder, f"{x}_labels.npy"))
    #             else np.nan
    #         )
    #     )

    #     self.metadata["embeddings_labels_support_size"] = self.metadata[f"embedding_labels"].apply(
    #         lambda x: len(np.unique(x)) if type(x) in [list, np.ndarray] else np.nan
    #     )
    #     self.metadata["n_embedding_labels"] = self.metadata[f"embedding_labels"].apply(
    #         lambda x: len(x) if type(x) in [list, np.ndarray] else np.nan
    #     )
    #     n = len(self.metadata)
    #     self.metadata.dropna(subset=["embedding_labels"], inplace=True)
    #     print(
    #         f"Discarding {n - len(self.metadata)} samples with missing embedding_labels. -> N={len(self.metadata)}."
    #     )
        
    #     # Add computation of the distance to the cluster centers
    #     cluster_centers = np.load(os.path.join(experiment_folder, 'cluster_centers.npy'))
    #     self.metadata['cluster_dist'] = self.metadata.apply(lambda x: np.linalg.norm(np.load(x.video_representation_path) - cluster_centers[x['embedding_labels']], axis=1), axis=1)

        

    def load_compute_hyperparameters(self, verbose=False):

        hyperparams_table_path = os.path.join(
            self.root_dir, "dataframes", "persistent_metadata", "hyperparams_table.csv"
        )
        hyperparams_table = pd.read_csv(hyperparams_table_path)

        self.metadata = self.metadata.merge(
            hyperparams_table[
                ["identifier", "delta_t", "segment_length", "num_frames"]
            ],
            on=["identifier"],
            how="left",
            indicator=True,
        ).rename(columns={"_merge": "hyperparameter_metadata_merge"})

        if verbose:
            pass#display(self.metadata["hyperparameter_metadata_merge"].value_counts())

        # Fix: backfill unique constant value
        self.metadata["segment_length"].fillna(16, inplace=True)
        self.metadata["segment_length"] = self.metadata["segment_length"].astype(int)
        self.metadata["delta_t"].fillna(8, inplace=True)
        self.metadata["delta_t"] = self.metadata["delta_t"].astype(int)
        self.metadata["token_duration"] = (
            self.metadata["segment_length"] / self.metadata["fps"]
        )

        # # We exclude for now the ebedding array that don't seem to match in terms of shape of X and expected shape of X (based on indices)
        # self.metadata["idx_embedding"] = self.metadata.apply(
        #     lambda x: (
        #         list(
        #             smartflat_range(
        #                 num_frames=int(x.n_frames),
        #                 segment_length=int(x.segment_length),
        #                 delta_t=int(x.delta_t),
        #             )
        #         )
        #         if not np.isnan(x.n_frames)
        #         else np.nan
        #     ),
        #     axis=1,
        # )

        # n = len(self.metadata)
        # # self.metadata.dropna(subset=["segment_length"], inplace=True)
        # print(
        #     "Discarding {} samples with missing hyperparameters. -> N={}.".format(
        #         n - len(self.metadata.dropna(subset=["segment_length"])),
        #         len(self.metadata),
        #     )
        # )

        # self.metadata["n_idx_embedding"] = self.metadata.idx_embedding.apply(
        #     lambda x: len(x) if not np.isnan(x).all() else np.nan
        # )

        self.metadata["has_hyperparams"] = self.metadata.num_frames.apply(
            lambda x: False if np.isnan(x) else True
        )

    def load_optimal_lambdas(self, config_name):

        config = import_config(config_name)

        optimal_lambdas_path = os.path.join(
            self.root_dir, "experiments", config.experiment_name, "lambda_optimal.csv"
        )
        if not os.path.isfile(optimal_lambdas_path):
            raise ValueError(
                f"No results of estimated regularization parameters found: {optimal_lambdas_path}."
            )

        print("Loading penalty parameters in {}.".format(optimal_lambdas_path))

        if config.model_params["penalty"] == "modality-specific":
            usecols = [
                "identifier",
                "L_hat^c",
                "k_hat^c",
                "x0_hat^c",
                "L_var^c",
                "k_var^c",
                "x0_var^c",
                "lambda_0^c",
                "lambda_1^c",
            ]

            subjects_df = pd.read_csv(optimal_lambdas_path, usecols=usecols)
            subjects_df.rename(
                columns={
                    "L_hat^c": "L_hat",
                    "k_hat^c": "k_hat",
                    "x0_hat^c": "x0_hat",
                    "L_var^c": "L_var",
                    "k_var^c": "k_var",
                    "x0_var^c": "x0_var",
                    "lambda_0^c": "lambda_0",
                    "lambda_1^c": "lambda_1",
                },
                inplace=True,
            )

        elif config.model_params["penalty"] == "personalized":
            usecols = [
                "identifier",
                "L_hat^p",
                "k_hat^p",
                "x0_hat^p",
                "L_var^p",
                "k_var^p",
                "x0_var^p",
                "lambda_0^p",
                "lambda_1^p",
            ]

            subjects_df = pd.read_csv(optimal_lambdas_path, usecols=usecols)
            subjects_df.rename(
                columns={
                    "L_hat^p": "L_hat",
                    "k_hat^p": "k_hat",
                    "x0_hat^p": "x0_hat",
                    "L_var^p": "L_var",
                    "k_var^p": "k_var",
                    "x0_var^p": "x0_var",
                    "lambda_0^p": "lambda_0",
                    "lambda_1^p": "lambda_1",
                },
                inplace=True,
            )
        else:
            raise ValueError(f'Unknown penalty type: {config.model_params["penalty"]}')

        self.metadata = self.metadata.merge(subjects_df, on="identifier", how="left")
        print(
            "Retrieve optimal lambda_0 and lambda_1 for {}/{} identifiers ({} mode).".format(
                len(self.metadata.dropna(subset=["lambda_0"])),
                self.metadata.identifier.nunique(), 
                config.model_params["penalty"],
            )
        )

class VideoSegmentDataset(VideoBlockDataset):
    """Dataset class for video segment representation.

    Args:
        change_point_experiment_id (str): Experiment ID containing the temporal segmentation result.
        agg_func (str): Aggregation function to use over the segment. One of ("middle", "mean", "random").

    Notes:
    - Use temporal segmentation outputs to index the passation-modality by segments instead.


    """

    def __init__(
        self,
        change_point_experiment_id,
        dataset_name="video_segment_representation",
        root_dir: str = None,
        task_names: list = ["cuisine", "lego"],
        modality: str = "all",
        scenario: str = None,
        normalization: str = "l2",
        n_pca_components: int = None,
        agg_func="middle",
    ):

        self.change_point_experiment_id = change_point_experiment_id
        self.agg_func = agg_func

        super().__init__(
            dataset_name=dataset_name,
            root_dir=root_dir,
            task_names=task_names,
            modality=modality,
            scenario=scenario,
            normalization=normalization,
            n_pca_components=n_pca_components,
        )

        self.segmentize()

    def __getitem__(self, index):
        meta_row = self.metadata.iloc[index]

        path = meta_row.video_representation_path
        labels = meta_row.groupe
        ids = meta_row.identifier

        inputs = self.transform(path)[meta_row.start : meta_row.end]
        inputs = self.aggregate_features(inputs)
        labels = self.target_transform(labels)
        return inputs, labels, ids

    def segmentize(self):
        """Expand dataset to be keyed by segments, as defined by the temporal segmentation."""

        self.load_change_point_detection(
            change_point_experiment_ids=[self.change_point_experiment_id]
        )

        add_cols = [
            "identifier",
            "task_name",
            "participant_id",
            "modality",
            "video_name",
            "video_representation_path",
            # "annotation",
            # "has_annotation",
            # "flag_video_representation",
            # "passation_id",
            # "fps",
            # "n_frames",
            # "date",
            # "duration",
            # "task_number",
            # "diag_number",
            # "trigram",
            "groupe",
            # "pathologie",
            # "ISDC",
            # "MoCA",
            # "groupe_bis",
            # "flag",
            # "n_cpts",
        ]

        df = []
        for i in range(len(self)):

            row = self.metadata.iloc[i]

            for s, e in pairwise(row["cpts"]):

                df_i = {}
                for add_col in add_cols:
                    df_i[add_col] = row[add_col]

                df_i.update({"start": s, "end": e})

                df.append(df_i)
        n = len(self.metadata)
        self.metadata = pd.DataFrame(df)
        print(
            f"Expanding dataset from {n} passsation-modality  to {len(self.metadata)} passation-modality-segments."
        )

    def aggregate_features(self, inputs):
        """Aggregate features over the segment."""

        if self.agg_func == "middle":
            N, _ = inputs.shape
            return inputs[N // 2].reshape(1, -1)

            # N, _ = inputs.shape
            # return inputs[N // 2 : N // 2 + 1]

        elif self.agg_func == "mean":
            return inputs.mean(axis=0)

        elif self.agg_func == "random":

            # N, _ = inputs.shape
            # idx = np.random.randint(N)
            # return inputs[idx : idx + 1]
            N, _ = inputs.shape
            idx = np.random.randint(N)
            return inputs[idx].reshape(1, -1)

        else:
            raise NotImplementedError(
                f"Aggregation function {self.agg_func} not implemented."
            )

class VideoSegmentRepresentationsDataset(VideoBlockDataset):
    """Dataset class for video segment representation, keyed by passation-modality.

    Notes:
    - Use temporal segmentation outputs to index the passation-modality by segments instead.
    """

    def __init__(
        self,
        dataset_name="video_segment_representation_unexpended",
        root_dir: str = None,
        task_names: list = ["cuisine", "lego"],
        modality: str = "all",
        scenario: str = None,
        normalization: str = "l2",
        n_pca_components: int = None,
        change_point_experiment_ids=ChangePointDetectionDeploymentConfig.experiment_ids,
        agg_func="middle",
    ):
        if modality:
            raise DeprecationWarning("Use VideoSegmentDataset instead.")

        if not isinstance(change_point_experiment_ids, list):
            change_point_experiment_ids = [change_point_experiment_ids]
        self.change_point_experiment_ids = change_point_experiment_ids
        self.agg_func = agg_func

        super().__init__(
            dataset_name=dataset_name,
            root_dir=root_dir,
            task_names=task_names,
            modality=modality,
            scenario=scenario,
            normalization=normalization,
            n_pca_components=n_pca_components,
        )

        # Load segmentation
        self.load_change_point_detection(
            change_point_experiment_ids=self.change_point_experiment_ids
        )

    def __getitem__(self, index):

        meta_row = self.metadata.iloc[index]

        path = meta_row.video_representation_path
        labels = meta_row.groupe
        ids = meta_row.identifier

        inputs = self.transform(path)
        inputs = np.vstack(
            [
                self.aggregate_features(inputs[start:end])
                for start, end in pairwise(meta_row["cpts"])
            ]
        )

        labels = self.target_transform(labels)
        return inputs, labels, ids

    def aggregate_features(self, inputs):
        """Aggregate features over the segment."""

        if self.agg_func == "middle":
            N, _ = inputs.shape
            return inputs[N // 2 : N // 2 + 1]

        elif self.agg_func == "mean":
            return inputs.mean(axis=0)

        elif self.agg_func == "random":

            N, _ = inputs.shape
            idx = np.random.randint(N)
            return inputs[idx : idx + 1]

        else:
            raise NotImplementedError(
                f"Aggregation function {self.agg_func} not implemented."
            )

if __name__ == "__main__":

    root_dir = get_data_root()

    print(f"Benchmarking dataset: '{VideoBlockDataset.__name__}'")
    test_dataset_video_representations(VideoBlockDataset, root_dir=root_dir)
    test_dataset_video_representations(VideoBlockDataset, root_dir=root_dir)
