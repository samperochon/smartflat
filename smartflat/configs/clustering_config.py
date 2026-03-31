"""Config API for the Smartflat pipeline."""

import os
import sys
from typing import Tuple



from smartflat.configs.smartflat_config import BaseSmartflatConfig
from smartflat.constants import available_modality, available_tasks

# 
# Final configs
#


class ClusteringBaseConfig(BaseSmartflatConfig):

    # --------  Change point detection model ---------

    experiment_name = "clustering"
    experiment_id = None  
    round_number: int  = None  # Set dynamically, Used to distinguish between different rounds of clustering
    annotator_id: str = None  # Set dynamically, Used to distinguish between different annotators
    model_name: str = "kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 20,
        "algorithm_name": "elkan",  # TODO: check if memory efficiency works well
        'normalization': 'Z-score' #TODO: for ow redudnat with the dataset level setting, but eventually will be more this one and the other set to idetntify 
    }
    
    task_type: str = "clustering"
    cpts_config_name: str ='ChangePointDetectionDeploymentConfig' 
    training_mode = True  # Set to False for inference on all samples

    task_name: str = "cuisine"
    modality: str = "Tobii" 
    n_sigma: int = 1
    thresholding_method: str = 'min_dist'  # 'min_dist' or 'threshold_mapping' or '1st' or 'median'
    threshold_value: float = 0.3  # Default value, can be overridden in specific configs
    
    # Shall we add all the symbolic configs attributes to make it feasible within the symbolic pipleine ? I don't think so. 
    # There are small domains overlap between symbolic and  clustering configs 
    # The idea is that experiemnts encapsulate in the clustering inferenc econfig are incorporated in symbolic configs (we keep all intermediary results) 

    #do_batch_whiten =  False # Introduce from Deployment, otherwise taken from the dataset_params
    do_add_timestamps = False # Introduce from Deployment, otherwise taken from the dataset_params
    n_clusters_dict = dict()
    
    input_clustering_config_name = None
    cluster_type =  None


##########################################################
# Preliminary step : Initial clustering (using all training samples)
##########################################################

class ClusteringDeploymentKmeansFullConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    # experiment_name = "gold-clustering-deployment-kmeans-all"
    # experiment_id = 'K_250' # Set in main_seployment 
    experiment_name = "symbolization-gold"
    experiment_id = 'init_clustering_cosine_full' # Set in main_seployment 
    
    
    n_clusters = [250]
    
    model_name: str = "batch-kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 250,
        "algorithm_name": "lloyd",  # TODO: check if memory efficiency works well
        'personalized': False,
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'],
        'kernel_name': 'cosine',
        'normalization': 'l2',
        'co_normalization': False, 
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False

class ClusteringDeploymentKmeansZscoreFullConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'init_clustering_euclidean_full' # Set in main_seployment 
    
    n_clusters = [250]
    
    model_name: str = "batch-kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 250,
        "algorithm_name": "lloyd",  # TODO: check if memory efficiency works well
        'personalized': False,
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'],
        'normalization': 'Z-score',
        'kernel_name': 'euclidean',
        'co_normalization': False, 
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False
    
# The two following faiss configs were used with consistent chnage of folder name, n_clusters, etc

##############################################################################
# Step 1: Initial clustering (Cross-validation using a 80/20 train/test split)
##############################################################################

class ClusteringDeploymentKmeansConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    # experiment_name = "gold-clustering-deployment-kmeans-all"
    # experiment_id = 'K_250' # Set in main_seployment 
    experiment_name = "symbolization-gold"
    experiment_id = 'init_clustering_cosine' # Set in main_seployment 
    
    n_clusters = [100]
    
    model_name: str = "batch-kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 100,
        "algorithm_name": "lloyd",  # TODO: check if memory efficiency works well
        'personalized': False,
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'],
        'kernel_name': 'cosine',
        'normalization': 'l2',
        'co_normalization': False, 
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False

class ClusteringDeploymentKmeansEuclideanConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'init_clustering_euclidean' # Set in main_seployment 

    n_clusters = [100]
    
    model_name: str = "batch-kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 100,
        "algorithm_name": "lloyd",  # TODO: check if memory efficiency works well
        'personalized': False,
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'],
        'normalization': 'Z-score',
        'kernel_name': 'euclidean',
        'co_normalization': False, 
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False
    
class ClusteringDeploymentFaissCKmeansConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'init_clustering_faissc' # Set in main_seployment 

    n_clusters = [100]
    
    model_name: str = "faiss-kmeans-cosine"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 100,
        "algorithm_name": "lloyd",  # TODO: check if memory efficiency works well
        'personalized': False,
        'task_name': 'cuisine',
        'modality': 'Tobii',
        
        #'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig'],
        'round_numbers': [1],  
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']],
        
        'cluster_types': ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'],
        'normalization': 'l2',
        'kernel_name': 'cosine',
        'co_normalization': False, 
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False

class ClusteringDeploymentFaissEKmeansConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'init_clustering_faisse' # Set in main_seployment 

    n_clusters = [100]
    
    model_name: str = "faiss-kmeans-euclidean"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 100,
        "algorithm_name": "lloyd",  # TODO: check if memory efficiency works well
        'personalized': False,
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'],
        'normalization': 'Z-score',
        'kernel_name': 'euclidean',
        'co_normalization': False, 
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False


##################################################################################
# Step 2: Inference post manual annotation (including best model + current 5 categories)
##################################################################################

class ClusteringDeploymentKmeansCombinationCompleteConfig(ClusteringBaseConfig):
    """Projection used after annotation is perfromed, for e.g. convergence rate estimate, or for analaysis"""
    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_post_hf'
    training_mode = False  # Set to False to avoid training mode in the deployment phase
    
    n_clusters = [250] # Placeholder
    
    # input_clustering_config_names stores the clustering configs for which we do manual annotation of the centroids
    # Then the qualification smapping store for each of them the clusters index in the reference of the test-time clustering (i.e accounting for
    # the prototypes of previous-stages and the new ones). As a sanity check, there should never be cluster index in the qualification 
    # mapping of downstream clustering configs indexes below the cumultive sum of all task/exogenous-specific clusters of the updtream configs
    model_name: str = "prototypes"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": None, # Set when pulling the centroids from the different clustering algorithms
        'personalized': False, #TODOSAM Check that this is good 
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig'],
        # 'round_numbers': [1],  
        # 'qualification_cluster_types': [ ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']],
        
        
        'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        'round_numbers': [1, 2],  
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']],

        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2, 3],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']],
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2, 3, 4],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']],
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 
        #                                    'SymbolicSourceInferenceRefinementGoldConfig', 
        #                                    'SymbolicSourceInferenceRefinementGoldConfig', 
        #                                    'SymbolicSourceInferenceRefinementGoldConfig', 
        #                                    'SymbolicSourceInferenceRefinementGoldConfig', 
        #                                    'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2, 3, 4, 5, 6],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']],
        
        #'input_clustering_config_names' : ['ClusteringDeploymentKmeansConfig', 'ClusteringDeploymentKmeansRefinementTaskConfig'],
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'], 
        #                                 ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']],
        
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'],

        'centroid_min_dist':  None, 
        'normalization': 'l2',
        'kernel_name': 'cosine',
        'normalize_distance_per_cluster': False,
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False


##################################################################################
# Step 3: Refinement procedure based on the inference
##################################################################################


class ClusteringDeploymentKmeansRefinementTaskConfig(ClusteringBaseConfig):
    """TODO: Complete
    
    ANNOTATION-TAG: annotation is perfromed on this config. 
    Prototypes in their own training space, without previously discovered well-defined prototypes. 
    
    """
    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_refinement' # Set in main_seployment 

    # experiment_name = "clustering-deployment-kmeans-task-r1"
    # experiment_id = 'K_250' # Set in main_seployment 
    n_clusters = [100]

    model_name: str = "faiss-kmeans-cosine"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 100,
        "algorithm_name": "elkan",  # TODO: check if memory efficiency works well
        'personalized': False,
        'kernel_name': 'cosine',
        'cluster_types': ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'],
        'normalization': 'l2',
        # 'input_clustering_config_names' : ['ClusteringDeploymentKmeansRefinementTaskConfig', 'ClusteringDeploymentKmeansRefinementTaskConfig', 'ClusteringDeploymentKmeansRefinementTaskConfig'],
        'round_numbers': [1, 2, 3, 4, 5, 6, 7],  
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']],
        
        'task_name': 'cuisine',
        'modality': 'Tobii',
        
        
        
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    
    # Input clustering definition, with a complete partitions of the centroids so far
    input_clustering_config_name = 'SymbolicSourceInferenceCompleteGoldConfig' #'ClusteringDeploymentKmeansCombinationCompleteConfig'
    thresholding_method = 'min_dist'  # 'min_dist' or 'threshold_mapping' or '1st' or 'median
    threshold_value = 0.3
    
    
    # # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    # do_batch_whiten =  False


##################################################################################
# Step 4 (Optional except final model): Inference with task/exo-definitive + noise 
##################################################################################

class ClusteringDeploymentKmeansInferenceConfig(ClusteringBaseConfig):

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_inference' # Set in main_seployment 
    
    
    n_clusters = [250] # Placeholder
    # input_clustering_config_names stores the clustering configs for which we do manual annotation of the centroids
    # Then the qualification smapping store for each of them the clusters index in the reference of the test-time clustering (i.e accounting for
    # the prototypes of previous-stages and the new ones). As a sanity check, there should never be cluster index in the qualification 
    # mapping of downstream clustering configs indexes below the cumultive sum of all task/exogenous-specific clusters of the updtream configs
    model_name: str = "prototypes"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": None, # Set when pulling the centroids from the different clustering algorithms
        'personalized': False, 
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig'],
        # 'round_numbers': [1],  
        # 'qualification_cluster_types': [ ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise']],

        'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        'round_numbers': [1, 2, 3, 4, 5],  
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],

                                        ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2, 3, 4],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'centroid_min_dist':  None, 
        'normalization': 'l2',
        'kernel_name': 'cosine',
        'normalize_distance_per_cluster': False,
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False


class ClusteringDeploymentKmeansMergeInferenceConfig(ClusteringBaseConfig):

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_merge_dinference' # Set in main_seployment 
    
    
    n_clusters = [250] # Placeholder
    # input_clustering_config_names stores the clustering configs for which we do manual annotation of the centroids
    # Then the qualification smapping store for each of them the clusters index in the reference of the test-time clustering (i.e accounting for
    # the prototypes of previous-stages and the new ones). As a sanity check, there should never be cluster index in the qualification 
    # mapping of downstream clustering configs indexes below the cumultive sum of all task/exogenous-specific clusters of the updtream configs
    model_name: str = "prototypes"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": None, # Set when pulling the centroids from the different clustering algorithms
        'personalized': False, 
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig'],
        # 'round_numbers': [1],  
        # 'qualification_cluster_types': [ ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise']],

        'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 
                                           'SymbolicSourceInferenceRefinementGoldConfig', 
                                           'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 

                                            'SymbolicSourceFaissCInferenceGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                            'SymbolicSourceInferenceRefinementGoldConfig', 
                                           'SymbolicSourceInferenceRefinementGoldConfig', 
                                           'SymbolicSourceInferenceRefinementGoldConfig'],
        'input_annotator_ids': ['samperochon', 'samperochon', 'samperochon', 'samperochon', 'samperochon', 'samperochon', 'samperochon', 'samperochon', 
                              'theoperochon', 'theoperochon', 'theoperochon', 'theoperochon', 'theoperochon', 'theoperochon', 'theoperochon', 'theoperochon'],
        'round_numbers': [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],  
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2, 3, 4],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'centroid_min_dist':  None, 
        'normalization': 'l2',
        'kernel_name': 'cosine',
        'normalize_distance_per_cluster': False,
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False


class ClusteringDeploymentKmeansDropFirstInferenceConfig(ClusteringBaseConfig):

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "symbolization-gold"
    experiment_id = 'faissc_inference_drop_first' # Set in main_seployment 
    
    
    n_clusters = [250] # Placeholder
    # input_clustering_config_names stores the clustering configs for which we do manual annotation of the centroids
    # Then the qualification smapping store for each of them the clusters index in the reference of the test-time clustering (i.e accounting for
    # the prototypes of previous-stages and the new ones). As a sanity check, there should never be cluster index in the qualification 
    # mapping of downstream clustering configs indexes below the cumultive sum of all task/exogenous-specific clusters of the updtream configs
    model_name: str = "prototypes"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": None, # Set when pulling the centroids from the different clustering algorithms
        'personalized': False, 
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig'],
        # 'round_numbers': [1],  
        # 'qualification_cluster_types': [ ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise']],

        'input_clustering_config_names' : [#'SymbolicSourceFaissCInferenceGoldConfig', 
                                           'SymbolicSourceInferenceRefinementGoldConfig',
                                           'SymbolicSourceInferenceRefinementGoldConfig', 
                                           'SymbolicSourceInferenceRefinementGoldConfig',

                                           'SymbolicSourceInferenceRefinementGoldConfig',
                                           'SymbolicSourceInferenceRefinementGoldConfig',
                                           'SymbolicSourceInferenceRefinementGoldConfig',
                                           'SymbolicSourceInferenceRefinementGoldConfig'],
        'round_numbers': [2, 3, 4, 5, 6, 7, 8],  
        'qualification_cluster_types': [
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        # 'input_clustering_config_names' : ['SymbolicSourceFaissCInferenceGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig', 'SymbolicSourceInferenceRefinementGoldConfig'],
        # 'round_numbers': [1, 2, 3, 4],  
        # 'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise'],
        #                                 ['task-definitive', 'exo-definitive', 'Noise']],
        
        
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'centroid_min_dist':  None, 
        'normalization': 'l2',
        'kernel_name': 'cosine',
        'normalize_distance_per_cluster': False,
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False




# (3.2) Inference using all the 250 training clusters (including ambiguous) with the already-collected prototypes 
class ClusteringDeploymentKmeansCombinationInferenceI1Config(ClusteringBaseConfig):

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "gold-clustering-deployment-prototypes-v1-inference"
    experiment_id = 'K_250' # Placeholder
    n_clusters = [250] # Placeholder
    # input_clustering_config_names stores the clustering configs for which we do manual annotation of the centroids
    # Then the qualification smapping store for each of them the clusters index in the reference of the test-time clustering (i.e accounting for
    # the prototypes of previous-stages and the new ones). As a sanity check, there should never be cluster index in the qualification 
    # mapping of downstream clustering configs indexes below the cumultive sum of all task/exogenous-specific clusters of the updtream configs
    model_name: str = "prototypes"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": None, # Set when pulling the centroids from the different clustering algorithms
        'personalized': False, 
        'input_clustering_config_names' : ['ClusteringDeploymentKmeansEuclideanConfig'], #SAM_MODEL_CHANGE 
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise']], # pLaceholder for not considering hte qualification mapping
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'centroid_min_dist':  None, 
        'normalization': 'l2',
        'kernel_name': 'euclidean',
        'normalize_distance_per_cluster': False,
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "Z-score", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False

class ClusteringDeploymentKmeansCombinationInferenceI5Config(ClusteringBaseConfig):
    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "clustering-deployment-prototypes-v5-inference-test"
    experiment_id = 'K_250' # Placeholder
    n_clusters = [250] # Placeholder
    # input_clustering_config_names stores the clustering configs for which we do manual annotation of the centroids
    # Then the qualification smapping store for each of them the clusters index in the reference of the test-time clustering (i.e accounting for
    # the prototypes of previous-stages and the new ones). As a sanity check, there should never be cluster index in the qualification 
    # mapping of downstream clustering configs indexes below the cumultive sum of all task/exogenous-specific clusters of the updtream configs
    model_name: str = "prototypes"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": None, # Set when pulling the centroids from the different clustering algorithms
        'personalized': False, 
        'input_clustering_config_names' : ['ClusteringDeploymentKmeansConfig',
                                           'ClusteringDeploymentKmeansRefinementTaskConfig', 
                                           'ClusteringDeploymentKmeansRefinementTaskI2Config', 
                                           'ClusteringDeploymentKmeansRefinementTaskI3Config',
                                           'ClusteringDeploymentKmeansRefinementTaskI4Config'], 
        
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'], 
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise'],
                                        ['task-definitive', 'exo-definitive', 'Noise']
                                        ],
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'cluster_types': ['task-definitive', 'exo-definitive', 'Noise'],
        'centroid_min_dist': None, 
        'normalization': 'Z-score',
        'kernel_name': 'euclidean',
        'normalize_distance_per_cluster': False,
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
    }
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False
    

# 3.1) Inference using definitive (+noise) initial and lately refined clusters
class ClusteringDeploymentKmeansCombinationTrainingConfig(ClusteringBaseConfig):
    """Training clusetring config, which does not appear in the qualification mapping 
     since  the annotation were perfromed on the refinement set this time (TODO: re-run all depending on the 
     experiment results of the next iteration)."""
    
    experiment_name = "clustering-deployment-prototypes-v1-training" #TODO: add a -training at the end
    experiment_id = 'K_placeholder' # PLACEHOLDER
    n_clusters = ['placeholder']
    # input_clustering_config_names stores the clustering configs for which we do manual annotation of the centroids
    # Then the qualification smapping store for each of them the clusters index in the reference of the test-time clustering (i.e accounting for
    # the prototypes of previous-stages and the new ones). As a sanity check, there should never be cluster index in the qualification 
    # mapping of downstream clustering configs indexes below the cumultive sum of all task/exogenous-specific clusters of the updtream configs
    model_name: str = "prototypes"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": None, # Set when pulling the centroids from the different clustering algorithms
        'personalized': False, 
        'input_clustering_config_names' : ['ClusteringDeploymentKmeansConfig', 'ClusteringDeploymentKmeansRefinementTaskConfig'],
        # Tris tells which type we take from which previous run (including by-passing )
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'], 
                          []], # pLaceholder for not considering hte qualification mapping and take the training model (K=250 typically) 
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'centroid_min_dist':  .3, 
        'normalization': 'Z-score',
        'cluster_types' : ['task-definitive', 'exo-definitive', 'Noise']
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False
    
class ClusteringDeploymentKmeansCombinationTrainingT2Config(ClusteringBaseConfig):
    """Training clusetring config, which does not appear in the qualification mapping 
     since  the annotation were perfromed on the refinement set this time (TODO: re-run all depending on the 
     experiment results of the next iteration)."""
    
    experiment_name = "clustering-deployment-prototypes-v2-training" #TODO: add a -training at the end
    experiment_id = 'K_placeholder' # PLACEHOLDER
    n_clusters = ['placeholder']
    # input_clustering_config_names stores the clustering configs for which we do manual annotation of the centroids
    # Then the qualification smapping store for each of them the clusters index in the reference of the test-time clustering (i.e accounting for
    # the prototypes of previous-stages and the new ones). As a sanity check, there should never be cluster index in the qualification 
    # mapping of downstream clustering configs indexes below the cumultive sum of all task/exogenous-specific clusters of the updtream configs
    model_name: str = "prototypes"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": None, # Set when pulling the centroids from the different clustering algorithms
        'personalized': False, 
        'input_clustering_config_names' : ['ClusteringDeploymentKmeansConfig', 
                                           'ClusteringDeploymentKmeansRefinementTaskConfig', 
                                           'ClusteringDeploymentKmeansRefinementTaskI2Config'],
        
        # Tris tells which type we take from which previous run (including by-passing )
        'qualification_cluster_types': [['task-definitive', 'exo-definitive', 'Noise'], 
                          ['task-definitive', 'exo-definitive', 'Noise'],
                          []], # pLaceholder for not considering hte qualification mapping and take the training model (K=250 typically) 
        'task_name': 'cuisine',
        'modality': 'Tobii',
        'centroid_min_dist':  .3, 
        'normalization': 'Z-score',
        'cluster_types' : ['task-definitive', 'exo-definitive', 'Noise']
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }

    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False



########################################################################################################
# Other clustering

class ClusteringDeploymentPersonalizedKmeansConfig(ClusteringBaseConfig):

    #Temp
    experiment_name = "clustering-deployment-kmeans-personalized"
    experiment_id = 'K_15'# Set in main_seployment 

    model_name: str = "kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 15,
        "algorithm_name": "elkan",  # TODO: check if memory efficiency works well
        'personalized': True
    }
    

    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": None,
        "n_pca_components": None,
        'whiten': None # Active only for normalization=PCA
    }
    do_batch_whiten =  False
    
class ClusteringDeploymentKmeansTimestampsConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "clustering-deployment-kmeans-timestamps-all"
    experiment_id = 'K_250' # Set in main_seployment 
    n_clusters = [250]

    model_name: str = "kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 15,
        "algorithm_name": "elkan",  # TODO: check if memory efficiency works well
        'personalized': False
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "Z-score", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False
    do_add_timestamps = True
    
class ClusteringDeploymentTWFINCHConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "clustering-deployment-twfinch-all"
    experiment_id = 'K_15' # Set in main_seployment 
    
    model_name: str = "tw-finch"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 15,
        'personalized': True
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False

class ClusteringDeploymentGaussianMixtureConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "clustering-deployment-gmm-all"
    experiment_id = 'K_15' # Set in main_deployment 
    
    model_name: str = "gaussian_mixture"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 15,
        'batch_size': 15000,
        'personalized': False # Define training set components
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False

class ClusteringDeploymentBayesianGaussianMixtureConfig(ClusteringBaseConfig):
    """
    #TODO: A bit hacky:
    Note: Typically, when clustering, the task and modality are changed when calling the clustering engine function. ."""

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "clustering-deployment-bayes-gmm-all"
    experiment_id = 'K_15' # Set in main_deployment 
    
    model_name: str = "bayesian_gaussian_mixture"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 15,
        'batch_size': 10000, 
        'personalized': False # Define training set components
    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None 
    }
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False



# Refineemnt configs 
class ClusteringDeploymentKmeansRefinementCoarseTaskConfig(ClusteringBaseConfig):

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "clustering-deployment-kmeans-task-r1-coarse"
    experiment_id = 'K_115' # Set in main_seployment 
    n_clusters = [115]

    model_name: str = "kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 115,
        "algorithm_name": "elkan",  # TODO: check if memory efficiency works well
        'personalized': False,
        'normalization': 'Z-score' #TODO: for ow redudnat with the dataset level setting, but eventually will be more this one and the other set to idetntify 

    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identify", #decide etween identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    
    # Input clustering definition, with a complete partitions of the centroids so far
    input_clustering_config_name = 'ClusteringDeploymentKmeansConfig' #'ClusteringDeploymentKmeansConfig'
    centroid_min_dist = 0.3
    
    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False
    
class ClusteringDeploymentKmeansRefinementExogeneousConfig(ClusteringBaseConfig):

    #TODO WAITING : Decide on definitive values based on the experiments 
    
    experiment_name = "clustering-deployment-kmeans-exogeneous-r1"
    experiment_id = 'K_250' # Set in main_
    n_clusters = [50]

    
    model_name: str = "kmeans"  #'pelt' or 'pelt_gram'
    model_params: dict = {
        "n_clusters": 250,
        "algorithm_name": "elkan",  # TODO: check if memory efficiency works well
        'personalized': False,
        'normalization': 'Z-score' #TODO: for ow redudnat with the dataset level setting, but eventually will be more this one and the other set to idetntify 

    }
    
    dataset_name: str = "video_block_representation"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity", #decide between identitif L1 or L2
        "n_pca_components": None, #decide 
        "whiten": None,
    }
    
    input_clustering_config_name = 'ClusteringDeploymentKmeansCombinationConfig'
    centroid_min_dist = 0.3

    # Fixed number of cluster for each modality and task (determined in experiment TODO) 
    do_batch_whiten =  False
    

#----------  Segments (keyed by segments)

class ClusteringSegmentsConfig(ClusteringBaseConfig):

    dataset_name: str = "video_segment_representation"
    experiment_name = "clustering-segments-200624"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": available_tasks,
        "modality": available_modality,
        "normalization": "identity",
        "n_pca_components": None,
        "agg_func": "middle",
        'change_point_experiment_id': 'lambda_1'
    }
    
class ClusteringSegmentsCuisineTobiiConfig(ClusteringBaseConfig):

    dataset_name: str = "video_segment_representation"
    experiment_name = "clustering-segments-200624"
    dataset_params: dict = {
        "root_dir": None,
        "scenario": "success",
        "task_names": ['cuisine'],
        "modality": ["Tobii"],
        "normalization": "identity",
        "n_pca_components": None,
        "agg_func": "middle",
        'change_point_experiment_id': 'lambda_1'
    }
    
