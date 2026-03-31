import inspect
import os
import sys
from typing import List

from smartflat import configs
from smartflat.configs.base_config import BaseConfig


def import_config(config_name: str) -> BaseConfig:
    """Import config by name from smartflat configs module."""
    config_class = getattr(configs, config_name)
    return config_class()


def load_config(config_filename: str) -> BaseConfig:
    """Load config from disk."""
    return BaseConfig.from_json(config_filename)


def get_complete_configs(round_number: int, inference_mode: bool=False):
    
    refinement_config_name = 'SymbolicSourceInferenceRefinementGoldConfig'

    if inference_mode:
        input_clustering_config_names = ['SymbolicSourceFaissCInferenceGoldConfig'] + [refinement_config_name] * (round_number-1)
        round_numbers = list(range(1, round_number + 1))
        qualification_cluster_types = [['task-definitive', 'exo-definitive', 'Noise']] * round_number
        
    else:
        input_clustering_config_names = ['SymbolicSourceFaissCInferenceGoldConfig'] + [refinement_config_name] * (round_number - 1)
        round_numbers = list(range(1, round_number + 1))
        qualification_cluster_types = [['task-definitive', 'exo-definitive', 'Noise']] * (round_number - 1)
        qualification_cluster_types.append(['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise'])
        
    return input_clustering_config_names, round_numbers, qualification_cluster_types
