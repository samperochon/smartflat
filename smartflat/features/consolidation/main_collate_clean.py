"""Clean up residual video partitions after successful collation.

When to run: After main_collate.py has merged video partitions into merged_video.mp4.
Prerequisites: Dataset with collate flag status; merged_video files present.
Outputs: Removes partition files and their feature outputs when collation succeeded.
Usage: python -m smartflat.features.consolidation.main_collate_clean
"""

import argparse
import os
import re
import socket
import subprocess
import sys

import numpy as np
import pandas as pd


from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_coding import red
from smartflat.utils.utils_collate import remove_output_files, write_collate
from smartflat.utils.utils_io import (
    get_data_root,
    get_file_size_in_gb,
    get_video_output_paths,
    parse_flag,
    parse_participant_id,
)


def extract_file_paths(file_path):
    paths = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"file '(.*)'", line)
            if match:
                paths.append(match.group(1))
    return paths

def clean_collate_residuals(dset, process=False):
    """Explore the Smartflat data in `root_dir` and remove video partitions (and outputs) if merged_video successful.
    
    TODO: Very similar to the function remove_output_files."""
    
    df = dset.metadata
    gdf = dset.global_metadata

    commands = []
    for (participant_id, modality), group  in df.groupby(['participant_id', 'modality']):

        df_merged = group[group['video_name'] == 'merged_video'].copy()
        df_merged_bar = group[group['video_name'] != 'merged_video'].copy()
        modality_folder = os.path.dirname(group.video_representation_path.iloc[0])
        # Compute size of the output video
        assert len(df_merged) < 2

        if len(df_merged) == 1:

            flag = parse_flag(df_merged.video_representation_path.iloc[0], modality, flag_type = 'flag_collate_video')
            if flag == 'success': # collate_flag may not have been transfered
                if len(df_merged_bar) > 1:
                    
                    for video_name, video_representation_path in df_merged_bar[['video_name', 'video_representation_path']].to_numpy():
                        red('What to add here ? ')
                        commands.extend([['rm', p] for p in get_video_output_paths(video_name, modality_folder)])
            elif flag == 'failure':
                print(f'Collate flag failure for {participant_id} and {modality}')
            
            else:
                print(f'Collate flag not found for {participant_id} and {modality}')

    if process:
        for command in commands:
            subprocess.run(command)
    else:
        for command in commands:
            print('\n'.join(command))
            
    return commands



def main(dset, process=False):
    """Check that the collate checks are correct wrt the global metadata
    
    Check on the collation of the merged video. This script is used to verify that the merged video is of correct size as
    found in the global metadata registration. 

    Process:
    We loop over the registered merged_video present in the machine. If the merged video is not here, we cannot check, and so the 
    the outputs registering the merged video are checked using another function TODO. 
    If the sizes don't match, we remove all outputs and video corresponding to the merge video. 
    
    """
        
    df = dset.metadata.copy()
    df = df[~((df['n_videos'] == 1) | (df['video_name'] == 'merged_video'))].copy()
    gdf = dset.global_metadata
    
    commands = []
    for (participant_id, modality), group  in df[df['video_name'] == 'merged_video'].groupby(['participant_id', 'modality']):
    
        # Compute size of the output video
        assert len(group) == 1
        merged_video_path = group.video_path.iloc[0]
        
        # TOFIX
        if participant_id == 'G11_P4_SAUJea_21032017':
            pass# see ?continue # Changed n_videos manualy as unsorted GoPro videos in Percy... better fix: change '?' in metadata modality 
        if (participant_id, modality) in [('G93_P79_AMEAmo_25052022', 'GoPro2')]:
            continue
    
        if group.video_path.isna().iloc[0]:
            print(f'Missing merged_video to compute size for {participant_id} and {modality} check aborted.')
            continue
    
        size_merged_video = get_file_size_in_gb(merged_video_path)
    
        # Compute source video sizes and check source videos metadata is correct
        cdf = gdf[ (gdf['participant_id'] == participant_id) & (gdf['modality'] == modality)]
        filepath = os.path.join(os.path.dirname(group['video_representation_path'].iloc[0]), 'collate_videos.txt')
        if not os.path.isfile(filepath):
            # Merged perform on another machine
            if modality != 'Tobii':
                # Process the videos that have standard names (8 characters)
                order = ['GOPR', 'GP01', 'GP02', 'GP03', 'GP04', 'GP05', 'GP06', 'GP07']
                cdf = cdf.assign(st=cdf.video_name.apply(lambda x: x[:4]))
                assert np.sum([0 if st in order else 1 for st in cdf.st]) == 0
                cdf['end'] = cdf.video_name.apply(lambda x: int(x[4:]))#.unique()  #[['participant_id', 'modality', 'video_name']]
                cdf['st_rank'] = cdf['st'].apply(lambda x: order.index(x) if x in order else len(order))
                cdf = cdf.sort_values(by=['participant_id', 'modality', 'st_rank', 'end'])
            else:
        
                cdf = cdf.sort_values(['participant_id', 'modality','order'])
                
            video_paths = [os.path.join(get_data_root(), group.task_name.iloc[0], participant_id, modality, video_name) for video_name in cdf.video_name.unique()]
            
            if process:
                write_collate(video_paths)
            else:
                print(f'Write missing collate ?   {video_paths}')

        if not os.path.isfile(filepath):
            print(f'Missing collate text {filepath} (video merged in another location)') #TODO: improve text file transfering ? Prb. not necessary for now
            continue
        source_video_paths = extract_file_paths(filepath)
        
        #TOFIX
        if participant_id == 'G100_P86_BAUVin_25112022' and modality == 'GoPro2':
            # For now we wait to dl the one in cheetah
            continue
        assert  cdf['n_videos'].iloc[0] == len(cdf) == len(source_video_paths) and len(cdf) > 1
        total_size = cdf['size'].sum()
    
        is_merged = np.isclose(total_size, size_merged_video, 0.05)
        if not is_merged:
            print(f"Report: {participant_id}-{modality} | n_videos: {cdf.n_videos.iloc[0]} | total size: {total_size} | merged size: {size_merged_video} | OKAY ?: {is_merged}")
            commands.extend(remove_output_files(group, process=process, include_videos_representations=True, include_videos=True))
            with open(os.path.join(os.path.dirname(merged_video_path), '.merged_video_collate_flag.txt'), 'w') as f:
                f.write('failure')
        else:
            with open(os.path.join(os.path.dirname(merged_video_path), '.merged_video_collate_flag.txt'), 'w') as f:
                f.write('success')
                
                
                
    print('Checking collate residuals (process=False)...')
    c_res = clean_collate_residuals(dset, process=False)
    for c in c_res:
        print(' '.join(c))
        
    return commands
                
                
    return commands


def parse_args():
    
    parser = argparse.ArgumentParser(description='Clean dataset consolidation and computation')
    parser.add_argument('-p', '--process', action="store_true", default=False, help='Process')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()    
    # Check on outputs and computation
    main(process=args.process)
    sys.exit(0)
