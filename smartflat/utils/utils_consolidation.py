import os
import subprocess
import sys

# ->TODO Add looking for the participant folder with incorrect structure (extra folders etc)
# -> TODO: /diskA/sam_data/data/cuisine/G102_P87_AUXCyr_09122022/GoPro1 has wrong merged_video block output, not the right size as the nexts videos. 


# add tools path and import our own tools

from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_io import collect_embeddings, fetch_flag_path


def extract_audio_files(process=False):
    '''Extract audio (.wav) files of consolidated videos.
    
    TODO: probably remove for avoiding circular imports with get_dataset...'''
    
    dset = get_dataset(dataset_name='base', scenario='present')
    df = dset.metadata.sort_values(['modality'], ascending=True).copy()
    df = df[(df['video_name'] == 'merged_video') | (df['n_videos'] == 1)]

    for video_path in df.video_path.tolist():
        
        output_audio_path = os.path.join(os.path.dirname(video_path), os.path.basename(video_path).split('.')[0] + '.wav')    
    
        if not os.path.isfile(output_audio_path) and process:
            subprocess.run(['ffmpeg', '-i', video_path, output_audio_path])
            print('Extracted audio file: {}.'.format(output_audio_path))

def get_unprocessed_videos():
    meta_unprocessed = []
    for output_type in ['video_block_representation', 'hand_landmarks', 'skeleton_landmarks', 'speech_recognition_representation']:
        
        dset = get_dataset(dataset_name=output_type, # 'base',  'video_block_representation', 'hands_landmarks', 'skeleton_landmarks', 'speech_recognition' 
                           scenario='unprocessed',  # 'all','success', 'unprocessed', 
                           task_names = 'all', # 'all', ['cuisine', 'lego']
                           modality='all' # 'all', ['GoPro1', 'GoPro2', 'GoPro3', 'Tobii']
                          )
    
        meta_unprocessed.append(dset.metadata.assign(output_type=output_type))
    
    unprocessed_videos = pd.concat(meta_unprocessed)
    return unprocessed_videos

def highlight_uncaught_videos():

    dset = get_dataset(dataset_name = 'base', scenario='all')
        
    df_all = dset.metadata.copy()
    
    present_videos = df_all[~df_all['video_path'].isna()].copy()
    unprocessed_videos = get_unprocessed_videos()
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
                                        'has_collate_video','flag_collate_video']])
    print('-------------- To be processed:')
    display(unprocessed_videos [ ['flag_speech_recognition', 'flag_speech_representation','flag_video_representation', 'flag_hand_landmarks','flag_skeleton_landmarks', 'flag_collate_video'] ] .apply(pd.Series.value_counts))    
    for output_type, group in unprocessed_videos.groupby(['output_type']):

        print(f'Unprocessed {output_type}:')
        print('\t\t\t\t\n'.join(group['video_path'].to_list()))
    
    return missed_videos_from_compute, unprocessed_videos

def copy_dataset_subset(df, output_folder, command_videos='cp', add_videos=False, process=True):
    """Takes as input a metadata dataframe and copy the subset of the dataset to an output folder.
    Copied data:
        1) Video if available
        2) Annotations
        3) Features extraction (hand_landmarks, speech_recognition, speech_representation, skeleton_landmarks, video_representation) 

    
    """
    
    print(f'Output folder: {output_folder}')

    # 1) Copy features extraction outputs and flags
    
    commands = []
    for output_type in [
        "video_representation",
        "hand_landmarks",
        "speech_recognition",
        "speech_representation",
        "skeleton_landmarks",
        ]:
        # commands = io.collect_embeddings(dset.metadata, output_root='/Volumes/Smartflat/data-gold' , output_type=output_type, separate_outputs=False, process=process)
        
        commands_output = collect_embeddings(
            df,
            output_root=output_folder,
            output_type=output_type,
            separate_outputs=False,
            process=process,
        )
        commands.extend(commands_output)


    # 2) Videos: Script that takes as input the metadata dataframe of a pre-defined dataset and copy the subset of the dataset to an output folder.

    commands = []
    for i, ((task_name, participant_id, modality), group) in enumerate(df.groupby(['task_name', 'participant_id', 'folder_modality'])):
        
        if not add_videos:
            continue
        
        video_paths = group.video_path.dropna().to_list()
        if len(video_paths) == 0:
            continue
        
        elif len(video_paths) > 1:
            red('More than one video path for a modality folder in a gold setting. Continue')
            print(video_paths)
            continue #raise ValueError('More than one video path for a modality folder in a gold setting.')
        
        
        output_folder_mod = os.path.join(output_folder, task_name, participant_id, modality); os.makedirs(output_folder_mod, exist_ok=True)
        video_path = group.video_path.unique()[0]
        try:
            output_video_path = os.path.join(output_folder_mod, os.path.basename(video_path))
        except:
            print(f'Error with {video_path}')
            continue
        
        if os.path.exists(output_video_path):
            print(f'Path {output_video_path} already exists')
            continue
        

        command = [command_videos, video_path, output_folder_mod]

        if process:


            try:
                blue(' '.join(command))
                subprocess.run(command)
            except:
                print('[ERROR]', command)
                display(group)
            
        else:
            try:
                yellow(' '.join(command))
            except:
                print('[ERROR]', command)
                display(group)
                # Type 1: participant folder which have been soft-changed to be fixed but still appear in the folders .. need to rsync and rm these wrong-named folders,
            commands.append(command)
    

    # 3) Sending annotations over

    commands = []
    for i, ((task_name, participant_id), group) in enumerate(df.groupby(['task_name', 'participant_id'])):
        
        participant_folder = os.path.join(output_folder, task_name, participant_id);
        if not os.path.isdir(participant_folder):
            print(f'Path {participant_folder} does not exist')
            continue
        
        os.makedirs(participant_folder, exist_ok=True)
        
        command = ['cp', '-r',  os.path.join(os.path.dirname(os.path.dirname(group.video_representation_path.iloc[0])), 'Annotation'), participant_folder]
        command = ['rsync', '-ahuvzL', '--progress',  os.path.join(os.path.dirname(os.path.dirname(group.video_representation_path.iloc[0])), 'Annotation'), participant_folder]


        if process:
            blue(' '.join(command))
            subprocess.run(command)

        else:
            yellow(' '.join(command))
            commands.append(command)
            
    return commands

