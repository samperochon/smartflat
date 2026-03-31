"""Config API for the Smartflat pipeline."""

import os
import sys
from typing import Tuple



from smartflat.configs.smartflat_config import BaseSmartflatConfig
from smartflat.constants import MEDIAN_SIGMA_RBF_N168, available_modality, available_tasks


class SymbolicBaseConfig(BaseSmartflatConfig):
    """Base Symbolic representaion configs."""


    experiment_name = None
    experiment_id = None  
    task_type: str = "symbolization"


    cpts_config_name: str = None #'ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = None #'ClusteringDeploymentKmeansCombinationInferenceI5Config'


    model_name: str = "prototypes" 
    model_params: dict = {}
    
    task_name: str = "cuisine"
    modality: str = "Tobii" 
    personalized: bool = False 
    #edges_opening_size =  30 # seconds
    #edges_closing_size = 2 # seconds    
    min_subjects = 2
    min_occurences = 5
    
    orthonormalization: bool = False
    use_segments_timetamps: bool = False
    timestamps_weighting_scheme: str = 'inverse_exp' # 'inverse_exp' or 'direct'
    temporal_distance: str  = 'wasserstein-1' # 'gw'
    perform_prototypes_reduction: bool = True

    temperature_tau: dict = {'K_space': 10,
                            'G_space': 10}
    temperature_x: dict =  {'K_space': 1,
                            'G_space': 1}
    agg_fn_str: str = 'mean' # 'mean', 'median', 'max', 'min'
    gridsize = 128
    loss_name: str = 'square_loss' # 'square_loss' or 'kl_loss'
    
    alpha: float = 1
    multimodal_method: str = 'multiplicative' # 'normalized_exp', 'unnormalized_exp', 'additive', 'multiplicative'
    # Overwrite dataset and model params(TODO: except prototypes thta for now is iteration-nromalized)
    #co_normalization: bool = True # 'identity' or 'Z-score', operates on samples and prototypes
    linkage_method: str = 'complete'
    
    
    radius_symbolization = None
    n_sigma = 1
    normalize_distance_per_cluster = False
    distance_aggregate_col = 'meta_cluster_min_dist'# Chosen between min, mean, median, max
    
    clustering_model_names: list = ['hierarchical-clustering']
    
    
    #do_batch_whiten =  False 
    do_add_timestamps = False 
    n_clusters_dict = dict()
    
    input_clustering_config_name = None
    cluster_type =  None
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 
    }    
    

# 
# Final Cropped configs
# 1) euclidean kernel
# 2) cosine kernel
# 3) rbf kernel



def fetch_cpts_config_name(is_cropped='cropped', kernel_name='euclidean', prototype_method='raw', calibrated=False):
    """Fetch the config name based on the parameters."""
    
    
    if prototype_method == 'latent':
        if kernel_name == 'euclidean':
            if is_cropped == 'cropped':
                if calibrated:
                    return 'ChangePointDetectionCroppedCalibratedDeploymentConfig'
                else:
                    return 'ChangePointDetectionCroppedDeploymentConfig'
            else:
                if calibrated:
                    return 'ChangePointDetectionCalibratedDeploymentConfig'
                else:
                    return 'ChangePointDetectionDeploymentConfig'
                
        elif kernel_name == 'cosine':
            if is_cropped == 'cropped':
                if calibrated:
                    return 'ChangePointDetectionCroppedCalibratedDeploymentConfig'
                else:
                    return 'ChangePointDetectionCroppedDeploymentConfig'
            else:
                if calibrated:
                    return 'ChangePointDetectionCalibratedDeploymentConfig'
                else:
                    return 'ChangePointDetectionDeploymentConfig'
                
class SymbolicSourceInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'prototypes_<kernel_name>_<prototypes_types>_<is_cropped>'  
    
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'
    is_cropped: bool = 'cropped'

    model_name: str = "prototypes" 
    model_params: dict = {
        'prototype_method': 'latent', 
        'inference_method': 'subspace_projection', 
        'kernel_name': None, # Overriden during experiments
        'sigma_rbf': None,
        'temperature': None, # Overriden during experiments
    }


class SymbolicPrototypesInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'prototypes_<kernel_name>_<prototypes_types>_<is_cropped>'  

    cpts_config_name: str ='ChangePointDetectionPrototypesCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    is_cropped: bool = 'cropped'

    model_name: str = "prototypes" 
    model_params: dict = {
        'prototype_method': None, # Overriden during experiments ['latent', 'raw', 'x_reduced_prototypes', 'xt_reduced_prototypes']
        'inference_method': 'subspace_projection', 
        'kernel_name': None, # Overriden during experiments ['cosine', 'euclidean', 'gaussian_rbf']
        'sigma_rbf': None,
        'temperature': None, # Overriden during experiments
    }


class SymbolicSourceCosineCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'rawspace_cosine_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionCroppedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': None,
        'use_K_space': False, # Use K-space for cosine kernel
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name': 'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', # "SymbolicSourceCosineInferenceConfig",
    }    
class SymbolicSourceEuclideanCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'rawspace_euclidean_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionCroppedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': None,
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', #"SymbolicSourceEuclideanInferenceConfig",
    }   
    
class SymbolicSourceGaussianRBFCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'rawspace_rbf_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionCroppedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'gaussian_rbf', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 0.5,
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name': 'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', # "SymbolicSourceGaussianRBFInferenceConfig",
    }    
    
    
#
# Prototypes subspace temporal segmentation
# 

class SymbolicPrototypesCosineCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'prototypes_cosine_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': None,

    }
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name':'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', # "SymbolicPrototypesCosineInferenceConfig",
    }    
    
class SymbolicPrototypesEuclideanCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'prototypes_euclidean_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': None,

    }
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name': 'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', # "SymbolicPrototypesEuclideanInferenceConfig",
    }    

class SymbolicPrototypesGaussianRBFCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'prototypes_rbf_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'gaussian_rbf', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 0.5,
    }
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name': 'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', # "SymbolicPrototypesGaussianRBFInferenceConfig",
    }    
#
# Reduced prototypes subspace temporal segmentation
# 

class SymbolicReducedPrototypesCosineCroppedImferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'reduced_prototypes_cosine_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , 
        'temperature': None,
}
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name': 'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', # "SymbolicReducedPrototypesCosineImferenceConfig",
    }    
    
class SymbolicReducedPrototypesEuclideanCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'reduced_prototypes_euclidean_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': None , 
        'temperature': None,

    }
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name': 'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', # "SymbolicReducedPrototypesEuclideanInferenceConfig",
    }    
    
class SymbolicReducedPrototypesGaussianRBFCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'reduced_prototypes_rbf_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'gaussian_rbf', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 0.5,

    }
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name': 'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', # "SymbolicReducedPrototypesGaussianRBFInferenceConfig",
    }    
    

# Additional testing configs

class SymbolicSourceIndepNormalizationEuclideanCroppedInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments-cropped"
    experiment_id = 'rawspace_euclidean_temporal_segmentation_indep_norm'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    co_normalization: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'clustering', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': None,
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": 'identity',
        "n_pca_components": None,
        'whiten': None, 
        #'crop_edges_config_name': 'SymbolicSourceIndepNormalizationEuclideanInferenceConfig',
    }

# 
# Final (non-cropped) )configs
#


#
# Source space temporal segmentation
# 

class SymbolicSourceCosineInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'rawspace_cosine_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 0.5,
    }


class SymbolicSourceEuclideanInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'rawspace_euclidean_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 0.5,
    }

class SymbolicSourceGaussianRBFInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'rawspace_rbf_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'gaussian_rbf', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 0.5,
    }

#
# Prototypes subspace temporal segmentation
# 

class SymbolicPrototypesCosineInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'prototypes_cosine_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': None,
    }


class SymbolicPrototypesEuclideanInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'prototypes_euclidean_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': None,
    }
    

class SymbolicPrototypesGaussianRBFInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'prototypes_rbf_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'gaussian_rbf', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 0.5,
    }

#
# Reduced prototypes subspace temporal segmentation
# 

class SymbolicReducedPrototypesCosineImferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'reduced_prototypes_cosine_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , 
        'temperature': None}


class SymbolicReducedPrototypesEuclideanInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'reduced_prototypes_euclidean_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': None , 
        'temperature': None,
    }
    

class SymbolicReducedPrototypesGaussianRBFInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'reduced_prototypes_rbf_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'gaussian_rbf', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 0.5,
    }



# 
# Additional testing configs
#
class SymbolicSourceIndepNormalizationEuclideanInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'rawspace_euclidean_temporal_segmentation_indep_norm'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    co_normalization: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'clustering', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': None,
    }
    

class SymbolicRawInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-v1"
    experiment_id = 'rawspace_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'personalized': False, 
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'normalization': 'Z-score',
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'orthonormalization': False, 
        'clustering_model_names': ['hierarchical-clustering'],
        'distance_aggregate_col': 'meta_cluster_min_dist',
        "radius_symbolization": None,
        "min_subjects": 2,
        "min_occurences": 2,
        'n_sigma': 1,
        'temperature': 0.5,
        'normalize_distance_per_cluster': False,

    }
    
class SymbolicClusteringInferenceConfig(SymbolicBaseConfig):
    """Symbolization with K-means-coming clustering (and reduced prototypes TS)"""
    
    experiment_name = "symbolization-v1"
    experiment_id = 'clustering_inference'  
    
    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'
    
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'clustering', 
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'personalized': False, 
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'normalization': 'Z-score',
        'kernel_name': 'euclidean', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'orthonormalization': False, 
        'clustering_model_names': ['hierarchical-clustering'],
        'distance_aggregate_col': 'meta_cluster_min_dist',
        "radius_symbolization": None,
        "min_subjects": 2,
        "min_occurences": 2,
        'n_sigma': 1,
        'temperature': 0.5,
        'normalize_distance_per_cluster': False,

    }

class SymbolicInferenceConfig(SymbolicBaseConfig):
    """Symbolization with prototypes projection and temporal segmentation in prototypes space"""

    experiment_name = "symbolization-v1"
    experiment_id = 'euclidean_subspace_projection'  

    cpts_config_name: str ='ChangePointDetectionPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'personalized': False, 
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'normalization': 'Z-score',
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'orthonormalization': False, 
        'clustering_model_names': ['hierarchical-clustering'],
        'distance_aggregate_col': 'meta_cluster_min_dist',
        "radius_symbolization": None,
        "min_subjects": 2,
        "min_occurences": 2,
        'n_sigma': 1,
        'temperature': 0.5,
        'normalize_distance_per_cluster': False,

    }

class SymbolicReduceInferenceConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the reduced prototypes space"""
    
    experiment_name = "symbolization-v1"
    experiment_id = 'reduced_prototypes'  
    
    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig'
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'personalized': False, 
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'normalization': 'Z-score',
        'inference_method': 'subspace_projection', 
        'kernel_name': 'euclidean', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'orthonormalization': False, 
        'clustering_model_names': ['hierarchical-clustering'],
        'distance_aggregate_col': 'meta_cluster_min_dist',
        "radius_symbolization": None,
        "min_subjects": 2,
        "min_occurences": 2,
        'n_sigma': 1,
        'temperature': 0.5,
        'normalize_distance_per_cluster': False,

    }



if __name__ == "__main__":

    config_names = ['SymbolicSourceCosineInferenceConfig', 
                    'SymbolicSourceEuclideanInferenceConfig', 
                    'SymbolicPrototypesGaussianRBFInferenceConfig', 
                    
                    'SymbolicPrototypesCosineInferenceConfig',
                    'SymbolicPrototypesEuclideanInferenceConfig',
                    'SymbolicPrototypesGaussianRBFInferenceConfig',
                    
                    'SymbolicReducedPrototypesCosineImferenceConfig',
                    'SymbolicReducedPrototypesEuclideanInferenceConfig',
                    'SymbolicReducedPrototypesGaussianRBFInferenceConfig',
                    
                    'SymbolicSourceIndepNormalizationEuclideanInferenceConfig'
                    
                    ]