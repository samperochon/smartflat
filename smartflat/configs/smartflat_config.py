"""Config API for the Smartflat pipeline."""

import os
import sys
from typing import List, Tuple


from smartflat.configs.base_config import BaseConfig


class BaseSmartflatConfig(BaseConfig):
    
    
    config_name = __name__
    
    
    # Dataset characteristics
    # available_tasks: List[str] = ['cuisine', 'lego']
    # available_modality: List[str] = ['Tobii', 'GoPro1', 'GoPro2', 'GoPro3']

    # expected_folders = ['GoPro1', 'GoPro2', 'GoPro3', 'Annotation', 'Tobii', 'Audacity']
    # modality_encoding = {'GoPro1': 0, 'GoPro2': 1, 'GoPro3': 2, 'Tobii': 3}
    
    # enabled_modalities = {'cuisine': {'flag_video_representation': ['GoPro1', 'GoPro2', 'GoPro3', 'Tobii'],
    #                             'flag_speech_recognition': ['GoPro1', 'GoPro2', 'GoPro3'],
    #                             'flag_speech_representation': ['GoPro1', 'GoPro2', 'GoPro3'],
    #                             'flag_hand_landmarks': ['GoPro2', 'Tobii'],
    #                             'flag_skeleton_landmarks': ['GoPro1']},
                    
    #                         'lego': {'flag_video_representation': ['GoPro1', 'GoPro2', 'GoPro3', 'Tobii'],
    #                                     'flag_speech_recognition': ['GoPro1', 'GoPro2', 'GoPro3'],
    #                                     'flag_speech_representation': ['GoPro1', 'GoPro2', 'GoPro3'],
    #                                     'flag_hand_landmarks': ['GoPro2', 'Tobii'],
    #                                     'flag_skeleton_landmarks': ['GoPro2']}
    #     }
    
            
    
    # --------  Video foundation model ---------
    # Frame per segments
    segment_length: int = 16
    # Elapsed number of frames between each temporal segments being represented
    delta_t: int  = 8 
    
    # --------  Pose landmarks estimation model ---------
    # Max number of body detected in a frame
    num_poses: int = 4 
    
    # --------  Hand landmarks estimation model ---------
    # Max number of hands detected in a frame
    num_hands: int = 4

    # --------- Change-point detection estimation --------
    #change_point_experiment_ids = ['Tobii_lambda_1', 'Tobii_lambda_2']

    join_len: int = .5  # [duration in seconds]

    # --------- Model identifiers ---------
    video_model_name: str = "vit_giant_patch14_224"
    speech_recognition_model_name: str = "whisperx"
    speech_embedding_model_name: str = "multilingual-e5-large"
    hand_landmarks_model_name: str = "hand_landmarks_mediapipe"
    skeleton_landmarks_model_name: str = "skeleton_landmarks_mediapipe"
    tracking_hand_landmarks_model_name: str = "tracking_hand_landmarks_v1"
