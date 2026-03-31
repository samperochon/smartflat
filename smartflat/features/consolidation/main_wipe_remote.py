"""Free up memory in the remote Cheetah location"""
import argparse
import os
import subprocess
import sys

import numpy as np


from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import get_data_root


def clean_videos_from_processed_folders(process=False):
    """Remove videos from processed folders"""
    dset = get_dataset(dataset_name='base', root_dir=get_data_root(), scenario='all')
    df = dset.metadata
    
    commands = []
    memory_saved = 0
    for _, group in df.groupby(['participant_id', 'modality']):


        if (group.processed == True).all():
    
            for video_path in group.video_path.dropna().tolist():
                commands.append(['rm', video_path])

            memory_saved += group.dropna(subset=['video_path'])['size'].sum()
    
    print("Removing {:.2f} Go of processed videos..".format(memory_saved))

    if process:
        for command in commands:
            subprocess.run(command)

    else:
        print('Turn process on to process: us.clean_videos_from_processed_folders(process=True)')

    return commands

def main(root_dir, process=False):
    """Remove videos from processed folders"""
    


    dset = get_dataset(dataset_name='base', root_dir=root_dir, scenario='all')
    df = dset.metadata
    
    commands = []
    memory_saved = 0
    for _, group in df.groupby(['participant_id', 'modality']):


        if (group.processed == True).all():
    
            for video_path in group.video_path.dropna().tolist():
                commands.append(['rm', video_path])

            memory_saved += group.dropna(subset=['video_path'])['size'].sum()
    
    print("Removing {:.2f} Go of processed videos..".format(memory_saved))

    if process:
        for command in commands:
            subprocess.run(command)

    else:
        print('Turn process on to process: us.clean_videos_from_processed_folders(process=True)')

    return commands

    
def parse_args():
    
    parser = argparse.ArgumentParser(description='Collate dataset')
    parser.add_argument('-p', '--process', action="store_true", default=False, help='Process the videos')
    parser.add_argument('-r', '--root_dir', default='local', help='local or harddrive')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
            
    if args.root_dir == 'local':
        
        root_dir = None
        
    elif args.root_dir == 'harddrive':
        
        root_dir = '/Volumes/harddrive/data' 
        
    main(root_dir = root_dir, process=args.process)
    exit(0)
