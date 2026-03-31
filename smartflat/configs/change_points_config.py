"""Config API for the Smartflat pipeline."""

import os
import sys
from typing import Tuple


from smartflat.configs.smartflat_config import BaseSmartflatConfig
from smartflat.constants import MEDIAN_SIGMA_RBF_N168, available_modality, available_tasks

TEMPERATURE = 1

class BaseChangePointDetectionConfig(BaseSmartflatConfig):
    """Default parameters."""
    
    # --------  Change point detection model ---------
    
    experiment_name = 'change-point-detection-base'
    experiment_id = None
    
    model_name: str = 'pelt' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'personalized',
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'cosine', 
                          'min_size': 5, 
                          'jump': 1
    }
    
    dataset_name: str = 'video_block_representation'
    dataset_params: dict = {'root_dir': None, 
                            'scenario': 'success',
                            'task_names': available_tasks,
                            'modality': available_modality,
                            'n_pca_components': None,
                            'normalization': 'Z-score'
                            }
    
    task_type: str = 'temporal_segmentation'
    whiten = False
    

# --------  Change point detection model ---------
# 1) Change point detection on the raw embedding space using the KernelCPD with vaious kernels
# 2) Change point detection on the prototypes space using the KernelCPD with vaious kernels
# 3) Change point detection on the reduced prototypes space 

# Note that this configs are lopped over with various kernel parameters and cropping versions

# Experiment configs to tune the penalty  parameters before deployment
class ChangePointDetectionExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    """
        
    experiment_name = "gold-change-point-detection-experiment"
    experiment_ids = None
    
    model_name: str = 'KernelCPD'
    model_params: dict = {'penalty': 'modality-specific', #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': 1.0, # rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None,
       # 'crop_edges_config_name': None,

    } 

class ChangePointDetectionPrototypesExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    
    
    Log determination sigma:
    DataFrame memory size: 76.211 MB
    DataFrame memory size: 0.086 MB
    [INFO] Shape of X: (1142746, 1408)
    Sigma=88.0728759765625 Gamma: 6.445930933394628e-05
    Kernel name: gaussian_rbf
    
    """
    experiment_name = "gold-change-point-detection-prototypes-experiment"
    experiment_ids = None
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    
    model_params: dict = {'penalty': 'modality-specific', #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
        "root_dir": None,
       # "config_name": 'SymbolicInferenceConfig',
        "scenario": "success",
        "task_names": 'cuisine',
        "modality": 'Tobii',
        "normalization": "identity",
        "n_pca_components": None,
        "whiten": None,
        "use_reduced_distance_matrix": True,
        'input_space': 'K_space'
       # 'crop_edges_config_name':  None,

    }
    
class ChangePointDetectionReducedPrototypesExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    
    
    Log determination sigma:
    DataFrame memory size: 76.211 MB
    DataFrame memory size: 0.086 MB
    [INFO] Shape of X: (1142746, 1408)
    Sigma=88.0728759765625 Gamma: 6.445930933394628e-05
    Kernel name: gaussian_rbf
    
    """
    experiment_name = "gold-change-point-detection-reduced-prototypes-experiment"
    experiment_ids = None
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty':  'modality-specific', #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
        "root_dir": None,
       # "config_name": 'SymbolicInferenceConfig',
        "scenario": "success",
        "task_names": 'cuisine',
        "modality": 'Tobii',
        "normalization": "identity",
        "n_pca_components": None,
        "whiten": None,
        "use_reduced_distance_matrix": True,
        'input_space': 'G_space'
        
       # 'crop_edges_config_name': None, 

    }


class KernelChangePointDetectionExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    """
        
    experiment_name = "gaussian-kernel-change-point-detection-experiment"
    experiment_ids = None
    
    model_name: str = 'KernelCPD'
    model_params: dict = {'penalty': 'calibration', #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 20, 
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': 1.0, # rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None,
       # 'crop_edges_config_name': None,

    } 

class KernelChangePointDetectionPrototypesExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    
    
    Log determination sigma:
    DataFrame memory size: 76.211 MB
    DataFrame memory size: 0.086 MB
    [INFO] Shape of X: (1142746, 1408)
    Sigma=88.0728759765625 Gamma: 6.445930933394628e-05
    Kernel name: gaussian_rbf
    
    """
    experiment_name = "kernel-change-point-detection-prototypes-experiment"
    experiment_ids = None
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    
    model_params: dict = {'penalty': 'calibration', #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                           'min_segment_duration': 20, 
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': 1.0, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
        "root_dir": None,
       # "config_name": 'SymbolicInferenceConfig',
        "scenario": "success",
        "task_names": 'cuisine',
        "modality": 'Tobii',
        "normalization": "identity",
        "n_pca_components": None,
        "whiten": None,
        "use_reduced_distance_matrix": True,
        'input_space': 'K_space'
       # 'crop_edges_config_name':  None,

    }
    
class KernelChangePointDetectionReducedPrototypesExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    
    
    Log determination sigma:
    DataFrame memory size: 76.211 MB
    DataFrame memory size: 0.086 MB
    [INFO] Shape of X: (1142746, 1408)
    Sigma=88.0728759765625 Gamma: 6.445930933394628e-05
    Kernel name: gaussian_rbf
    
    """
    experiment_name = "kernel-change-point-detection-reduced-prototypes-experiment"
    experiment_ids = None
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty':  'calibration', #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                        'min_segment_duration': 20, 
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': 1.0, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
        "root_dir": None,
       # "config_name": 'SymbolicInferenceConfig',
        "scenario": "success",
        "task_names": 'cuisine',
        "modality": 'Tobii',
        "normalization": "identity",
        "n_pca_components": None,
        "whiten": None,
        "use_reduced_distance_matrix": True,
        'input_space': 'G_space'
        
       # 'crop_edges_config_name': None, 

    }





# Individualized penalty parameter 
class ChangePointDetectionDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "gold-change-point-detection-deployment"
    experiment_ids = ['lambda_0', 'lambda_1']

    model_name: str = 'KernelCPD'
    model_params: dict = {'penalty': 'personalized', 
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': 1.0, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None,
       # 'crop_edges_config_name':  None,
    }

class ChangePointDetectionPrototypesDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "gold-change-point-detection-prototypes-deployment"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'personalized', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # When model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": False,
           # 'crop_edges_config_name':  None, 
        }

class ChangePointDetectionReducedPrototypesDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "change-point-detection-reduced-prototypes-deployment"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'personalized', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": True,
           # 'crop_edges_config_name':  None, 

        }


# Calibrated with unique penalty parameter 
class ChangePointDetectionCalibratedDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the modality-specific estimated penalty values.
    """
        
    experiment_name = "gold-change-point-detection-deployment-calibrated"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD'
    model_params: dict = {'penalty': 'modality-specific', 
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': 1.0, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None,
      # 'crop_edges_config_name':  None,

    }

class ChangePointDetectionPrototypesCalibratedDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the modality-specific estimated penalty values.
    """
        
    experiment_name = "gold-change-point-detection-prototypes-deployment-calibrated"

    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'modality-specific', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": False,
           # 'crop_edges_config_name':  None, 
        }

class ChangePointDetectionReducedPrototypesCalibratedDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the modality-specific estimated penalty values.
    """
        
    experiment_name = "change-point-detection-reduced-prototypes-deployment-calibrated"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'modality-specific', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": True,
           # 'crop_edges_config_name':  None, 

        }


# Kernel-based estimation of the temporal segmentation using the slope heuristics
class KernelChangePointDetectionDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "gaussian-kernel-change-point-detection-deployment"
    experiment_ids = ['gaussian_KCP']

    model_name: str = 'KernelCPD'
    model_params: dict = {'penalty': 'slope_heuristics',
                          'global_slope_heuristics':  -9.746668895818161e-05,
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 20,
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': 1.0, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ['Tobii'],
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None,
       # 'crop_edges_config_name':  None,
    }

class KernelChangePointDetectionPrototypesDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "kernel-change-point-detection-prototypes-deployment"
    experiment_ids = ['gaussian_KCP']
    
    
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'personalized', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # When model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": False,
           # 'crop_edges_config_name':  None, 
        }

class KernelChangePointDetectionReducedPrototypesDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "kernel-point-detection-reduced-prototypes-deployment"
    experiment_ids = ['gaussian_KCP']
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'personalized', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": True,
           # 'crop_edges_config_name':  None, 

        }




# --------  Change point detection model for cropped datasets ---------

# 1) Change point detection on the raw embedding space using the KernelCPD with vaious kernels
# 2) Change point detection on the prototypes space using the KernelCPD with vaious kernels
# 3) Change point detection on the reduced prototypes space 

class ChangePointDetectionCroppedExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    """
        
    experiment_name = "change-point-detection-experiment-cropped"
    experiment_ids = None
    
    model_name: str = 'KernelCPD'
    model_params: dict = {'penalty': None, #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None,
       # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 
    }

class ChangePointDetectionCroppedPrototypesExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    
    Log determination sigma:
    DataFrame memory size: 76.211 MB
    DataFrame memory size: 0.086 MB
    [INFO] Shape of X: (1142746, 1408)
    Sigma=88.0728759765625 Gamma: 6.445930933394628e-05
    Kernel name: gaussian_rbf
    
    """
    experiment_name = "change-point-detection-prototypes-experiment-cropped"
    experiment_ids = None
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    
    model_params: dict = {'penalty': None, #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
        "root_dir": None,
        "config_name": 'SymbolicInferenceConfig',
        "scenario": "success",
        "task_names": 'cuisine',
        "modality": 'Tobii',
        "normalization": "identity",
        "n_pca_components": None,
        "whiten": None,
        "use_reduced_distance_matrix": False,
       # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 

    }
    
class ChangePointDetectionCroppedReducedPrototypesExperimentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    
    
    Log determination sigma:
    DataFrame memory size: 76.211 MB
    DataFrame memory size: 0.086 MB
    [INFO] Shape of X: (1142746, 1408)
    Sigma=88.0728759765625 Gamma: 6.445930933394628e-05
    Kernel name: gaussian_rbf
    
    """
    experiment_name = "change-point-detection-reduced-prototypes-experiment-cropped"
    experiment_ids = None
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': None, #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
        "root_dir": None,
        "config_name": 'SymbolicInferenceConfig',
        "scenario": "success",
        "task_names": 'cuisine',
        "modality": 'Tobii',
        "normalization": "identity",
        "n_pca_components": None,
        "whiten": None,
        "use_reduced_distance_matrix": True,
       # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 
    }

        
class ChangePointDetectionCroppedDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "change-point-detection-deployment-cropped"
    experiment_ids = ['lambda_0', 'lambda_1']

    model_name: str = 'KernelCPD'
    model_params: dict = {'penalty': 'personalized', 
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None,
       # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 

    }
        
class ChangePointDetectionCroppedPrototypesDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "change-point-detection-prototypes-deployment-cropped"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'personalized', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # When model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": False,
           # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 
        }

class ChangePointDetectionCroppedReducedPrototypesDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the `personalized`` estimated penalty values.
    """
        
    experiment_name = "change-point-detection-reduced-prototypes-deployment-cropped"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'personalized', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": True,
           # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 

        }


class ChangePointDetectionCroppedCalibratedDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the modality-specific estimated penalty values.
    """
        
    experiment_name = "change-point-detection-deployment-calibrated-cropped"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD'
    model_params: dict = {'penalty': 'modality-specific', 
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None,
      # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 

    }

class ChangePointDetectionCroppedPrototypesCalibratedDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the modality-specific estimated penalty values.
    """
        
    experiment_name = "change-point-detection-prototypes-deployment-calibrated-cropped"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'modality-specific', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": False,
           # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 
        }
    
class ChangePointDetectionCroppedReducedPrototypesCalibratedDeploymentConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all modality and tasks 
    using the modality-specific estimated penalty values.
    """
        
    experiment_name = "change-point-detection-reduced-prototypes-deployment-calibrated-cropped"
    experiment_ids = ['lambda_0', 'lambda_1']
    
    model_name: str = 'KernelCPD' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 'modality-specific', #Overwriten at runtime for each task and modality
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'rbf', 
                          'sigma_rbf': None, # computed as the median euclidean distance between Z-normalized prototypes and smaple matrix
                          'temperature': TEMPERATURE, # 0.1 for rbf kernel
                          'min_size': 5, 
                          'jump': 1,
                          'norm_used': 'l2', # 2hen model name is pelt
    }
    
    dataset_name: str = "prototypes_trajectory_representation"
    dataset_params: dict = {
            "root_dir": None,
            "config_name": 'SymbolicInferenceConfig',
            "scenario": "success",
            "task_names": 'cuisine',
            "modality": 'Tobii',
            "normalization": "identity",
            "n_pca_components": None,
            "whiten": None,
            "use_reduced_distance_matrix": True,
           # 'crop_edges_config_name':  'SymbolicSourceIndepNormalizationEuclideanInferenceConfig', 

        }

    
    
    
    
    
class ChangePointDetectionAllConfig(BaseChangePointDetectionConfig):
    
    # --------  Change point detection model ---------
    experiment_name = 'change-point-detection-final'
    dataset_name: str = 'video_block_representation'
    dataset_params: dict = {'root_dir': None, 
                            'scenario': 'success',
                            'task_names': available_tasks,
                            'modality': available_modality,
                            'normalization': 'Z-score'}

class ChangePointDetectionExperimentHighMinConfig(BaseChangePointDetectionConfig):
    """
    Config to run the CPD on all block embeddings using varying penalty parameter
    and create the CPD-curves from which are estimated parameters (A, k, lambda_0)
    and both modality-specific and personalized optimal penalty values. 
    """
        
    experiment_name = "change-point-detection-experiment-high-min"
    experiment_ids = None
    
    model_params: dict = {'penalty': None, #Overwriten at runtime for each task and modality
                          'num_samples_penalty': 25, 
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'cosine', 
                          'min_size': 15, 
                          'jump': 1
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "Z-score",
        "n_pca_components": None,
        "whiten": None 
    }



    
class ChangePointDetectionPCAAllConfig(BaseSmartflatConfig):
    # --------  Change point detection model ---------
    
    experiment_name = 'change-point-detection-pca'
    experiment_id = None
    model_name: str = 'pelt' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 10,
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'cosine', 
                          'min_size': 5, 
                          'jump': 1
    }
    
    dataset_name: str = 'video_block_representation'
    dataset_params: dict = {'root_dir': None, 
                            'scenario': 'success',
                            'task_names': available_tasks,
                            'modality': available_modality,
                            'n_pca_components': 200,
                            'normalization': 'PCA-Z-score'}
    
    task_type: str = 'temporal_segmentation'
        
class ChangePointDetectionPCAWhitenConfig(BaseSmartflatConfig):
    # --------  Change point detection model ---------
    
    experiment_name = 'change-point-detection-pca-whiten-dev'
    experiment_id = None
    
    model_name: str = 'pelt' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 10,
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'cosine', 
                          'min_size': 5, 
                          'jump': 1
    }
    
    dataset_name: str = 'video_block_representation'
    dataset_params: dict = {'root_dir': None, 
                            'scenario': 'success',
                            'task_names': ['cuisine'],
                            'modality': ['Tobii'],
                            'n_pca_components': 200,
                            'normalization': 'PCA',
                            'whiten': True}
    
    task_type: str = 'temporal_segmentation'
    whiten = True
    #assert not ((task_type == 'temporal_segmentation') and 
    #        (dataset_params['normalization'] == 'PCA') and 
    #        (whiten==False))    TODO Check always the case (and add cases..)
      
         
# TODO
class ChangePointDetectionGramConfig(BaseSmartflatConfig):
    # --------  Change point detection model ---------

    model_name: str = 'pelt_gram' #'pelt' or 'pelt_gram'
    model_params: dict = {'penalty': 10,
                          'min_segment_duration': 1, # 'seconds'
                          'kernel': 'linear', 
                          'min_size': 5, 
                          'jump': 1
    }
    
    dataset_name: str = 'video_block_representation'
    dataset_params: dict = {'root_dir': None, 
                            'scenario': 'success',
                            'task_names': available_tasks,
                            'modality': ['Tobii']}
    
    