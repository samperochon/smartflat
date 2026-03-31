"""Config API for the Smartflat pipeline."""

import os
import sys
from typing import Tuple



from smartflat.configs.symbolic_config import SymbolicBaseConfig
from smartflat.constants import MEDIAN_SIGMA_RBF_N168, available_modality, available_tasks

##########################################################
# Preliminary step : Initial clustering (using all training samples)
##########################################################

class SymbolicSourceCosineInferenceGoldFullConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'full_cosine_iteration_1'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansFullConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }

class SymbolicSourceEuclideanInferenceGoldFullConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'full_euclidean_iteration_1'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansZscoreFullConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }

class SymbolicSourceFaissCInferenceGoldFullConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'full_faissc_iteration_1'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentFaissCKmeansConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }

class SymbolicSourceFaissEInferenceGoldFullConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'full_faisse_iteration_1'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentFaissEKmeansConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }
    
    
##############################################################################
# Step 1: Initial clustering (Cross-validation using a 80/20 train/test split)
##############################################################################

class SymbolicSourceCosineInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'cosine_iteration_1'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None ,  # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }

class SymbolicSourceEuclideanInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'euclidean_iteration_1'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansEuclideanConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }
    
class SymbolicSourceFaissCInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_iteration_1'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentFaissCKmeansConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }
    
class SymbolicSourceFaissEInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faisse_iteration_1'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentFaissEKmeansConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }


##################################################################################
# Step 2: Inference post manual annotation (including best model + current 5 categories)
##################################################################################

class SymbolicSourceInferenceCompleteGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_post_hf_symbolization'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationCompleteConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }


##################################################################################
# Step 3: Refinement procedure based on the inference
##################################################################################

class SymbolicSourceInferenceRefinementGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_refinement_symbolization'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansRefinementTaskConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }

##################################################################################
# Step 4: (Optional except final model): Inference with task/exo-definitive + noise 
##################################################################################

class SymbolicSourceInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_inference_symbolization'  

    cpts_config_name: str ='KernelChangePointDetectionDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansInferenceConfig'

    perform_prototypes_reduction: bool = True
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
        'task_name': 'cuisine', 
        'modality': 'Tobii', 
        'use_K_space': True, # Use K-space for cosine kernel
    }

class SymbolicSourceMergeInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_merge_inference_symbolization'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansMergeInferenceConfig'

    perform_prototypes_reduction: bool = True
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
    
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
        'task_name': 'cuisine', 
        'modality': 'Tobii', 
        'use_K_space': True, # Use K-space for cosine kernel
    }


class SymbolicSourcePrototypesTSInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_pts_inference_symbolization'  

    cpts_config_name: str ='ChangePointDetectionPrototypesCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansInferenceConfig'

    perform_prototypes_reduction: bool = True
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
        'task_name': 'cuisine', 
        'modality': 'Tobii', 
        'use_K_space': True, # Use K-space for cosine kernel
    }
class SymbolicSourceReducedPrototypesTSInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_rpts_inference_symbolization'  

    cpts_config_name: str ='ChangePointDetectionReducedPrototypesCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansInferenceConfig'

    perform_prototypes_reduction: bool = True
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
        'task_name': 'cuisine', 
        'modality': 'Tobii', 
        'use_K_space': True, # Use K-space for cosine kernel
    }


class SymbolicSourceInferenceFropFirstGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_inference_drop_first_symbolization'  

    cpts_config_name: str ='ChangePointDetectionCalibratedDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansDropFirstInferenceConfig'

    perform_prototypes_reduction: bool = False
    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
        'task_name': 'cuisine', 
        'modality': 'Tobii', 
        'use_K_space': True, # Use K-space for cosine kernel
    }
#
# Prototypes subspace temporal segmentation
# 
class SymbolicPrototypesGaussianRBFInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'prototypes_rbf_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': None , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }

#
# Reduced prototypes subspace temporal segmentation
# 
class SymbolicReducedPrototypesGaussianRBFInferenceGoldConfig(SymbolicBaseConfig):
    """Symbolization with reduced prototypes projection and temporal segmentation in the raw space"""

    experiment_name = "symbolization-experiments"
    experiment_id = 'reduced_prototypes_rbf_temporal_segmentation'  

    cpts_config_name: str ='ChangePointDetectionReducedPrototypesDeploymentConfig' 
    clustering_config_name: str = 'ClusteringDeploymentKmeansCombinationInferenceI5Config'

    model_name: str = "prototypes" 
    model_params: dict = {
        'inference_method': 'subspace_projection', 
        'kernel_name': 'cosine', 
        'sigma_rbf': MEDIAN_SIGMA_RBF_N168 , # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
        'temperature': 1.0,
    }

