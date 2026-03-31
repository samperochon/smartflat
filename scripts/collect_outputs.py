import sys
import os
import socket

import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns

sys.path.insert(0, "/gpfs/users/perochons/smartflat")

from api.utils.utils_io import get_data_root, get_free_space
from api.datasets.loader import get_dataset
from api.datasets.base_dataset import SmartflatDataset
import api.utils.utils_io as io
from api.datasets.loader import get_dataset

dset = get_dataset(dataset_name = 'base', scenario = 'all')

process=True
for output_type in [ 'video_representation',  'speech_recognition', 'speech_representation','hand_landmarks', 'skeleton_landmarks']:
    io.collect_embeddings(dset.metadata, output_type=output_type, process=process)
    