import ast
import datetime
import os
import socket
import sys
from copy import deepcopy
from glob import glob
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# add tools path and import our own tools


import random

import matplotlib.pyplot as plt
from decord import VideoReader, bridge, cpu

from smartflat.annotation_smartflat import AnnotationSmartflat
from smartflat.constants import (
    available_output_type,
    mapping_incorrect_modality_name,
    mapping_participant_id_fix,
    tasks_duration_lims,
)
from smartflat.utils.utils import check_and_convert
from smartflat.utils.utils_clinical import diagnosis_logic
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_io import (
    check_video,
    fetch_flag_path,
    fetch_output_path,
    get_data_root,
    get_file_size_in_gb,
    get_video_loader,
    parse_dir_name,
    parse_flag,
    parse_participant_id,
    parse_path,
    parse_task_number,
)

bridge.set_bridge('native')

def show_random_frame(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_idx = random.randint(0, len(vr) - 1)
    frame = vr[frame_idx].asnumpy()
    plt.imshow(frame)
    plt.axis('off')
    plt.show()
    
def show_video_metadata(df_video):
    def p5(x):
        return x.quantile(0.05)

    def p10(x):
        return x.quantile(0.1)

    def p90(x):
        return x.quantile(0.9)

    def p95(x):
        return x.quantile(0.95)

    # Calculate descriptive statistics
    stats = df_video.groupby(["task_name", "modality"]).agg(
        {
            "participant_id": ["nunique"],
            "video_path": ["size", "count"],
            "fps": ["size", "count", p5, "median", "mean", p95, "std", "nunique"],
            "n_frames": ["size", "count", p5, "median", "mean", p95, "std", "nunique"],
            "size": ["size", "count", p5, "median", "mean", p95, "std", "nunique"],
        }
    )
    return stats

