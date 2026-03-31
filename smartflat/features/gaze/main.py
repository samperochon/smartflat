"""Hndle extracted Tobii gaze files.


Sam Perochon, adapted from https://github.com/OpenGVLab/VideoMAEv2. December 2024.

"""
import argparse
import math
import os
import random
import socket
import sys
import time

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import numpy as np
import pandas as pd
from tqdm import tqdm



from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import (
    fetch_output_path,
    get_api_root,
    get_data_root,
    get_free_space,
    get_host_name,
)


raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--cuda',
        default='0',
        type=str,
        help='GPU id to use')

    parser.add_argument('--chunk_idx', type=int, default=0, help='Index of the chunk to process')
    parser.add_argument('--num_chunks', type=int, default=10, help='Number of chunks to split the workload into')
    parser.add_argument('-r', '--do_reversed', action="store_true", default=False, help='Reverse the list of video paths to process.')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    main(device, num_chunks=args.num_chunks, chunk_idx=args.chunk_idx, do_reversed=args.do_reversed)
