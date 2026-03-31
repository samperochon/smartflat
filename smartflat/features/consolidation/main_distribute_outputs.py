import argparse
import os
import subprocess
import sys
from glob import glob

import numpy as np


from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_io import (
    check_exist,
    collect_embeddings,
    fetch_path,
    get_data_root,
    get_file_size_in_gb,
    parse_path,
)


def main(input_folder, output_dir=None, process=False, overwrite=False):
    """Copy the content of a source embedding folder to the machine data_root, creating subfolders if necessary.
    Works by iterating over the folder of the provided directory (e.g. cuisine, lego, sub-partitions, etc), then participant_id, then modality, then for each files present there (except videos), the embeddings are 

    Notes: Set `prompt_user` to False to execute the command without prompting the user to take action.

    Usage: Script: copy A to B, conserving the existing structure (of the data organisation).

    `
    import smartflat.utils.utils_io as io

    input_metafolder = os.path.join(get_data_root(), 'dump')
    copy_file(input_metafolder)
    `
    TODO: Make the function more general to be used with arbitrary folders sub-directories list."""
        
        
    n_copied = 0
    commands = []

    dump_folders = [path for path in glob(os.path.join(input_folder, '*')) if os.path.isdir(path)]

    for dump_folder in dump_folders:
        tasks_paths = [path for path in glob(os.path.join(dump_folder, '*')) if os.path.isdir(path)]

        for task_path in tasks_paths:

            participant_paths = [path for path in glob(os.path.join(task_path, '*')) if os.path.isdir(path)]

            for participant_path in participant_paths:

                modality_paths = [path for path in glob(os.path.join(participant_path, '*')) if os.path.isdir(path)]

                for modality_path in modality_paths:

                    # apply function to the embedding folder
                    content_paths = [path for path in glob(os.path.join(modality_path, '*'))] + [path for path in glob(os.path.join(modality_path, '.*'))]
                    
                    for content_path in content_paths:

                        try:
                            file_name, modality, participant_id, task =  parse_path(content_path)
                        except ValueError:
                            print('Parsed information: ', file_name, modality, participant_id, task)
                            print(f'Error parsing path: {content_path}')
                            continue
                        
                        output_path = fetch_path(task, participant_id, modality, output_dir=output_dir); os.makedirs(output_path, exist_ok=True)  
                        
                        
                        if os.path.exists(os.path.join(output_path, os.path.basename(content_path))):
                            #print(f"{os.path.join(output_path, os.path.basename(content_path))} exists. Overwrite={overwrite}")                        
                            # Command to be executed from a source embedding folder to a source embedding folder
                            if overwrite:
                                commands.append(['cp',  content_path, output_path]); n_copied+=1
                                
                        else:
                            commands.append(['cp',  content_path, output_path]); n_copied+=1
                            
                                            
    print(f'Process is {process} (N={n_copied}): ')
    if process:
        for command in commands:
            blue(' '.join(command))
            subprocess.run(command)
    else:
        for command in commands:
            yellow(' '.join(command))
        
    return commands 
    
    
def copy_extracted_features(output_dir="/Volumes/Smartflat/data-features", separate_outputs=False, process=False): 
    """Copy extracted features of all modalities to a new directory."""
    dset = get_dataset(
        dataset_name="base",  # 'base',  'video_block_representation', 'hands_landmarks', 'skeleton_landmarks', 'speech_recognition'
        scenario="all",  # 'all','success', 'unprocessed',
        task_names="all",  # 'all', ['cuisine', 'lego']
        modality="all",  # 'all', ['GoPro1', 'GoPro2', 'GoPro3', 'Tobii']
    )

    for output_type in [
        "hand_landmarks",
        "speech_recognition",
        "speech_representation",
        "skeleton_landmarks",
        "video_representation",
    ]:
        commands = collect_embeddings(
            dset.metadata,
            output_root=output_dir,  # "/diskA/sam_data/data-gold",
            output_type=output_type,
            separate_outputs=separate_outputs,
            process=process,
        )
    return commands



def parse_args():
    
    parser = argparse.ArgumentParser(description='Collate dataset')
    parser.add_argument('-p', '--process', action="store_true", default=False, help='Process the videos')
    parser.add_argument('-i', '--input_folder', default='local', help='local or path to the dump folder.')
    parser.add_argument('-o', '--output_dir', default='local', help='local or path to the target data folder.')

    return parser.parse_args()

if __name__ == '__main__':
        
    args = parse_args()
            
    if args.input_folder == 'local':
        
        input_folder = os.path.join(get_data_root(), 'dump')
        
    elif os.path.exists(args.input_folder):
        
        input_folder = args.input_folder
        
    else:
        raise ValueError(f"Path {args.input_folder} does not exist.")
    

    if args.output_dir == 'local':
        
        output_dir = os.path.join(get_data_root())
        
    elif os.path.exists(args.output_dir):
        
        output_dir = args.output_dir
        
    else:
        raise ValueError(f"Path {args.output_dir} does not exist.")
    
    main(input_folder = input_folder, output_dir=output_dir, process=args.process)
    
    print(f"Done distributing features from {args.input_folder} to {args.output_dir}")
    sys.exit(0)
    
