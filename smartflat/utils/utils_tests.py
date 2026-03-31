import datetime
import json
import logging
import os
import pickle
import socket
import subprocess
import sys
from glob import glob

import matplotlib.pyplot as plt

#TODO: weird that the three are imported only for assign_computation, maybe change the location...
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

# add tools path and import our own tools

from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_collate import remove_output_files
from smartflat.utils.utils_io import (
    fetch_flag_path,
    get_data_root,
    get_file_size_in_gb,
    parse_flag,
)


def open_folders(commands, n=5):
    for command in commands[:n]:
        print(command[1])
        subprocess.run(['open', os.path.dirname(command[1])] )
    
    
def check_video_outputs_size(dset=None, df=None, process=False):
    
    if dset is not None:
        df = dset.metadata.copy()
    
    commands = []

    df['size_bis'] = df['video_path'].apply(lambda x: get_file_size_in_gb(x) if type(x) == str else np.nan)
    for video_path in df[df['size_bis'] == 0].video_path.to_list():
        commands.append(['rm', video_path])
    print(f'Removing {len(df[df.size_bis == 0])} vieo of empty size')
    
    for feature in  ['speech_recognition', 'speech_representation','hand_landmarks', 'skeleton_landmarks', 'video_representation']:
        
        df[f'size_{feature}'] = df.apply(lambda x: get_file_size_in_gb(x[f'{feature}_path']) if x[f'{feature}_computed'] else np.nan, axis=1)
        
        corrupted_outputs = df[df[f'size_{feature}'] == 0][f'{feature}_path'].to_list()
        corrupted_outputs_flags = df[df[f'size_{feature}'] == 0][f'flag_{feature}'].to_list()
        corrupted_flags = [fetch_flag_path(p, f'flag_{feature}') for p in corrupted_outputs]
    
        print(f'Feature {feature} has {len(corrupted_outputs)} outputs of zero size.')
        
        
        for p in corrupted_outputs:
            commands.append(['rm',p])
        for p in corrupted_flags:
            #commands.append(['rm', p])
            with open(p, 'w') as f:
                f.write('failure')
            print('Wrote failure flag in ', p)
            
            
        plt.figure()
        df[f'size_{feature}'].hist(bins=50)
    
        plt.title('Feature {} memory footprint (GB; all={}'.format(feature, df[f'size_{feature}'].sum().round(1)))
        
        
    if  process:
        for command in commands:
            subprocess.run(command)
    else:
        for command in commands:
            print(' '.join(command))
    return commands

def test_output_flag_failed_but_present(dset, process = False):
    
    """Test for each output whether the output is present and the flag different than sucess (cleaning prupose)
    
    Look for video that may have a success but no files, for each outputs
    
    """
    
    df = dset.metadata.copy()
    
    commands = []
    for output_type in ['video_representation', 'hand_landmarks', 'skeleton_landmarks']:
    
        df_with_failed_present_output = df[ (df[f'{output_type}_computed']) & (df[f'flag_{output_type}'] != 'success') ]
        if len(df_with_failed_present_output) > 0:
            for output_path in df_with_failed_present_output[f'{output_type}_path'].to_list():
                if process:
                    os.remove(output_path)
                    os.remove(fetch_flag_path(output_path, f'flag_{output_type}'))
                else:
                    commands.append(['rm', output_path])
                    commands.append(['rm', fetch_flag_path(output_path, f'flag_{output_type}')])
                    
    return commands

def check_computed_video_block_embedding(dset, process=False):

    """TODO: check failed video with check_video"""

    df = dset.metadata.copy()
    df = df[df['video_representation_computed']]
    
    for i, row in df[(df.flag_video_representation == 'failure') & (df.video_representation_computed)].iterrows():
        print(row.video_representation_path)
        print("Shape of a video embedding that failed but computed: ", np.load(row.video_representation_path).shape)
        print(row.video_representation_path)
    
    
    n = 0
    n_tot = len(df)
    commands =[]
    for i, row in df.iterrows():
        
        try:
            is_null = np.load(row['video_representation_path']).mean() == 0
        except ValueError:
            print(f'Error loading {row.video_representation_path}')
            if process:
                print(f'Writing failure flag in {os.path.join(os.path.dirname(row.video_representation_path), f".{row.video_name}_video_representation_flag.txt")}')
                with open(os.path.join(os.path.dirname(row.video_representation_path), f'.{row.video_name}_video_representation_flag.txt'), 'w') as f:
                    f.write('failure')
        
        
        if is_null:
            print("{} is null".format(row.video_representation_path))
            commands.append(['rm', row['video_representation_path']])
            n+=1
            if process:
                print(f'Writing failure flag in {os.path.join(os.path.dirname(row.video_representation_path), f".{row.video_name}_video_representation_flag.txt")}')
                with open(os.path.join(os.path.dirname(row.video_representation_path), f'.{row.video_name}_video_representation_flag.txt'), 'w') as f:
                    f.write('failure')
                try:
                    os.remove(os.path.join(os.path.dirname(row.video_representation_path), f'.{row.video_name}_video_representation_flag.txt'))
                    os.remove(row.video_representation_path)
                    print(f'Wrote failure flag in {os.path.join(os.path.dirname(row.video_representation_path), f".{row.video_name}_video_representation_flag.txt")}')
                except:
                    pass
        else:
            # Success flag
            pass#if process:
                #print(f'Writing success flag in {os.path.join(os.path.dirname(row.video_representation_path), f".{row.video_name}_video_representation_flag.txt")}')
                #with open(os.path.join(os.path.dirname(row.video_representation_path), f'.{row.video_name}_video_representation_flag.txt'), 'w') as f:
                #    f.write('success')
    
    return commands 
    
    
def fix_unflagged_outputs(dset, process=False):
    
    df =  dset.metadata.copy()
    
    for output_name, output_type in zip(['video_representation',  'speech_recognition', 'speech_representation','hand_landmarks', 'skeleton_landmarks'],
                                        ['np', 'json', 'np', 'json', 'json']):
    
        print(f'Retrieve unflagged output for {output_name}...')
        for i, row in df[(df[f'{output_name}_computed']) & (df[f'flag_{output_name}'] == 'unprocessed')].iterrows():
    
            if output_type == 'np':
                print(np.load(row[f'{output_name}_path']))
            else:
                print('unflagged output: ', row[f'{output_name}_path'])
                with open(row[f'{output_name}_path'], 'r') as f:
                    try:
                        data = json.load(f)
                    except:
                        print(f'Error loading {row[f"{output_name}_path"]}')
                        continue
                #print(data)
        
            if process: 
                
                is_success = input('Does the output looks okay ? y or n')

                flag_path = fetch_flag_path(row[f'{output_name}_path'], f'flag_{output_name}')
                print(f'Writing flag in {flag_path}')
                
                if is_success:
                    with open(flag_path, 'w') as f:
                        f.write('success')
                else:
                    with open(flag_path, 'w') as f:
                        f.write('failure')

        print('Done.')

def wipe_partition_files(dset, process=False):
    """Task - Remove partition source and outputs"""
        
    df = dset.metadata.copy()
    
    df['has_video_representations_merge'] = df.apply(lambda x: (x.n_videos == 1) or os.path.isfile(os.path.join(dset.root_dir, x.task_name, x.participant_id, x.modality, 'video_representations_VideoMAEv2_merged_video.npy')), axis=1)
    
    commands = []
    for i, group in df.groupby(['participant_id', 'modality']):
    
    
        # if (group.n_videos == 1).all():
        #     if len(group) != 1:
        #         display(group)
        #     continue
    
        # if (group.processed == False).all():
        #     pass

        # if not  (group.n_videos.nunique() == 1):
        #     print(f'NOT UNIQUE N VIDEOS {group.n_videos.unique()}')
        #     display(group)
        
        if len(group) > 1:
            print(f'PARTITION {group.participant_id.iloc[0]} {group.modality.iloc[0]}')
            display(group)
        
        elif len(group) == 1:
            continue
        
        
        # 1) Clean partitions files if collated
        if (group.has_merged_video == True).all():
    
    
            # Extra video files ? 
            if len(group) > 1:
    
                extradf = group[group['video_name'] != 'merged_video']
    
                for file_exists, hand_landmarks_path in zip(extradf.hand_landmarks_computed.tolist(), extradf.hand_landmarks_path.tolist()):
                    if file_exists:
                        
                        commands.append(['rm', hand_landmarks_path])
                        commands.append(['rm', fetch_flag_path(hand_landmarks_path, 'flag_hand_landmarks')])
    
                for file_exists, skeleton_landmarks_path in zip(extradf.skeleton_landmarks_computed.tolist(), extradf.skeleton_landmarks_path.tolist()):
                    if file_exists:
                        commands.append(['rm', skeleton_landmarks_path])
                        commands.append(['rm', fetch_flag_path(skeleton_landmarks_path, 'flag_skeleton_landmarks')])
    
    
                for file_exists, speech_recognition_path in zip(extradf.speech_recognition_computed.tolist(), extradf.speech_recognition_path.tolist()):
                    if file_exists:
                        commands.append(['rm', speech_recognition_path])
                        commands.append(['rm', fetch_flag_path(speech_recognition_path, 'flag_speech_recognition')])
    
                
                for file_exists, speech_representation_path in zip(extradf.speech_representation_computed.tolist(), extradf.speech_representation_path.tolist()):
                    if file_exists:
                        commands.append(['rm', speech_representation_path])
                        commands.append(['rm', fetch_flag_path(speech_representation_path, 'flag_speech_representation')])
    
                if (group.has_video_representations_merge == True).all():
    
                    if group[group['video_name'] == 'merged_video'].flag_video_representation.iloc[0] == 'success':
                        for video_representation_path in extradf.video_representation_path.tolist():
                            
                            if os.path.isfile(video_representation_path):
                                
                                commands.append(['rm', video_representation_path])
                                commands.append(['rm', fetch_flag_path(video_representation_path, 'flag_video_representation')])
                    else:
                        print("[WARNING MODALITY FOLDER] unsuccessful merged_video video block embedding but partitions block embedding files")

        else:
            yellow('Without collate video')
            display(group)
            commands.extend(remove_output_files(group, process=process, include_videos_representations=True, include_videos=True))

    if process:
        for command in commands: 
            subprocess.run(command)
    return commands

def remove_valdity_outliers(dset, process=False):
    """Delete video and outputs of the videos/modality spotted as outliers by the manual visual inspection"""
    
    print('[WARNING] You have to toggle the visual inspection results befor eusing this function, otherwise the dataframe genration remove the outliers')
    df = dset.metadata.copy()
    
    cols = ['participant_id', 'modality', 'is_fish_eyed', 'upside_down', 'is_swapped_modality', 'true_modality', 'GP3_is_sink', 'GP2_is_wrong_buttress', 
       'GP2_above', 'GoPro1_is_wrong_buttress', 'is_middle_range', 'true_task_name', 'is_old_setup', 'is_old_recipe', 'annot_notes']
    results =  pd.read_csv(os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', 'results_visual_inspection.csv'), sep=';', usecols=cols)
        
    df = df.merge(results, on=['participant_id', 'modality'], how='left', indicator=False)    
    df['is_validity_outlier'] = False
    df.loc[~((df['upside_down'] != 1)
            & (df['true_task_name'] != np.nan)
            & (df['GP3_is_sink'] != 1)
            & (df['GP2_is_wrong_buttress'] != 1)
            & (df['GP2_above'] != 1)
            & (df['GoPro1_is_wrong_buttress'] != 1)
            & (df['GP3_is_sink'] != 1)
            & (df['is_middle_range'] != 1)
            & (df['GP3_is_sink'] != 1)
            & (df['GP3_is_sink'] != 1)), 'is_validity_outlier'] = True

    commands = []
    
    for video_path in df[df.is_validity_outlier == True].video_path.dropna():
        commands.append(['rm', video_path])
    
    if process:
        for command in commands:
            subprocess.run(command)

    return commands


def get_unprocessed_videos(dataset_name='base', root_dir=None):
    
    meta_unprocessed = []
    for output_type in ['video_block_representation', 'hand_landmarks', 'skeleton_landmarks', 'speech_recognition_representation']:
        
        dset = get_dataset(dataset_name=output_type, # 'base',  'video_block_representation', 'hands_landmarks', 'skeleton_landmarks', 'speech_recognition' 
                           root_dir=root_dir, 
                           scenario='unprocessed',  # 'all','success', 'unprocessed', 
                           task_names = 'all', # 'all', ['cuisine', 'lego']
                           modality='all' # 'all', ['GoPro1', 'GoPro2', 'GoPro3', 'Tobii']
                          )
    
        meta_unprocessed.append(dset.metadata.assign(output_type=output_type))
    
    unprocessed_videos = pd.concat(meta_unprocessed)
    return unprocessed_videos

def highlight_uncaught_videos(dataset_name='base', root_dir = None):
    """Highlight the state of the videos in a dta aroot. Note that the `processed` videos are characterize as uncaught currently (so wiping the machine is first required)."""

    dset = get_dataset(dataset_name=dataset_name, 
                       root_dir=root_dir,
                       scenario='all')
        
    df_all = dset.metadata.copy()
    
    present_videos = df_all[~df_all['video_path'].isna()].copy()
    unprocessed_videos = get_unprocessed_videos(dataset_name=dataset_name, root_dir=root_dir)
    present_videos = df_all[~df_all['video_path'].isna()].copy()
    present_videos = present_videos.merge(unprocessed_videos.drop_duplicates(subset=['identifier'])[['identifier']], on='identifier', how='left', indicator=True)
    
    present_videos['_merge'].replace({'left_only':'uncaught videos', 'both': 'videos remaining to process'}).hist()
    
    # Take videos that are present but uncaught from any compute (casper)
    missed_videos_from_compute = present_videos[present_videos['_merge'] == 'left_only']
    print('-------------- Uncaught videos:')

    display(missed_videos_from_compute[['identifier', 'video_name', 'video_path', 'n_videos', 'processed', 
                                        'speech_recognition_computed', 'flag_speech_recognition',
                                        'speech_representation_computed','flag_speech_representation',
                                        'video_representation_computed','flag_video_representation', 
                                        'hand_landmarks_computed','flag_hand_landmarks',
                                        'skeleton_landmarks_computed','flag_skeleton_landmarks', 
                                        'has_merged_video','flag_collate_video']])
    print('-------------- To be processed:')
    display(unprocessed_videos [ ['flag_speech_recognition', 'flag_speech_representation','flag_video_representation', 'flag_hand_landmarks','flag_skeleton_landmarks', 'flag_collate_video'] ] .apply(pd.Series.value_counts))    
    for output_type, group in unprocessed_videos.groupby(['output_type']):

        print(f'Unprocessed {output_type}:')
        print('\t\t\t\t\n'.join(group['video_path'].to_list()))
    
    return missed_videos_from_compute, unprocessed_videos


def open_wrong_flag(dset, process=False):
    
    expected_flag_values = ['success', 'failure', 'disabled', 'unprocessed']

    for output_flag in ['flag_speech_recognition',	'flag_speech_representation', 'flag_video_representation', 'flag_hand_landmarks', 'flag_skeleton_landmarks', 'flag_collate_video']:
        
        for path in dset.metadata[~dset.metadata[output_flag].isin(expected_flag_values)].video_representation_path.to_list():
            
            if process:
                subprocess.run(['open', os.path.dirname(path)])
            else:
                print(path)
                
                