"""Extract audio (.wav) from consolidated video files.

When to run: After video collation (main_collate.py) has produced merged videos.
Prerequisites: Dataset with consolidated videos (merged_video or single-video administrations).
Outputs: .wav audio files alongside each video in the modality folder.
Usage: python -m smartflat.features.consolidation.main_audio_extraction
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import signal


from smartflat.constants import (
    mapping_incorrect_modality_name,
    modality_encoding,
    video_extensions,
)
from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import extract_audio


def main(root_dir=None):
    """Extract(audio) .wav files from consolidated videos. """
        
    dset = get_dataset(dataset_name='base', root_dir=root_dir, scenario='present')
    df = dset.metadata; df = df[(df['video_name'] == 'merged_video') | (df['n_videos'] == 1)]
    for video_path in df.video_path.tolist():
            extract_audio(video_path)
    print('Done')
    

def parse_args():
    
    parser = argparse.ArgumentParser(description='Audio extraction')
    parser.add_argument('-r', '--root_dir', default='local', help='local or harddrive or Smartflat')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
            
    if args.root_dir == 'local':
        
        root_dir = None
        
    elif args.root_dir == 'harddrive':
        
        root_dir = '/Volumes/harddrive/data' 
        
    elif args.root_dir == 'Smartflat':
        
        root_dir = '/Volumes/Smartflat/data' 
        
    main(root_dir = root_dir)
    
    sys.exit(0)