'''Register Percy metadata'''
import datetime
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd


from smartflat.constants import mapping_incorrect_modality_name, root_paths, video_extensions
from smartflat.datasets.corrections import apply_manual_fixes
from smartflat.features.consolidation.main_snapshot import main as main_snapshot
from smartflat.utils.utils_io import check_video, get_data_root, get_file_size_in_gb


def main(path_metadata=None):
    '''Register all videos of multiple root paths in a single metadata file.'''
    
    #
    # 1) Inspect the different locations hunting for any videos 
    #
    videos_dict  = {} 
    for root_path in root_paths:
        found=False
        n = 0
        for dirpath, dirnames, filenames in os.walk(root_path):

            for filename in filenames:

                if any(filename.lower().endswith(ext) for ext in video_extensions):

                    found=True
                    n+=1

                    path = os.path.join(dirpath, filename)
                    timestamp = os.path.getctime(os.path.join(dirpath, filename))
                    date = datetime.datetime.fromtimestamp(timestamp)
                    size = get_file_size_in_gb(os.path.join(dirpath, filename))

                    video = cv2.VideoCapture(path)
                    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video.get(cv2.CAP_PROP_FPS)

                    videos_dict[path] = {'root_path': root_path,
                                         'video_name': filename.split('.')[0],
                                         'date': date,
                                         'size': size,
                                         'n_frames': n_frames,
                                         'fps': fps
                                        }

        print(f'{root_path}: found {n} videos.')


    #
    # 2) Post-processing
    #
    
    metadata = pd.DataFrame(videos_dict).transpose().reset_index().rename(columns={'index': 'video_path'})
    # Save dataframe 
    metadata.to_csv(path_metadata, index=False)
    print('{} videos registered - total size: {} To'.format(metadata.shape[0], metadata.size.sum()/1024))
    
    
    # Remove outlier videos 
    metadata = metadata[metadata['size'] > 0.05]
    metadata = metadata[metadata['n_frames'] > 50]
    metadata = metadata[metadata['fps'] > 4]


    metadata = metadata.assign(participant_id = metadata.video_path.apply(lambda x: [x.replace(root_path+"\\", '').split('\\')[0] for root_path in root_paths if x.startswith(root_path) ][0]),
                               video_name = metadata.video_path.apply(lambda x: [x.replace(root_path+"\\", '').split('\\')[-1][:-4] for root_path in root_paths if x.startswith(root_path) ][0]),
                               task_name = metadata.root_path.apply(lambda x: 'lego' if 'lego' in x.lower() else 'cuisine' if 'gateau' in x.lower() else np.nan ), 
                               duration = metadata.n_frames/metadata.fps,
                               modality = metadata.video_path.apply(lambda x: x.split('\\')[-2]).map(mapping_incorrect_modality_name) # /!\ Sensible to metadata created with Windows-style path. TOFIX
                              )
    
    metadata.loc[(metadata['video_name'] == 'fullstream'), 'modality'] = 'Tobii'


    metadata['date'] = pd.to_datetime(metadata['date'])
    metadata['video_name'] = metadata['video_name'].str.strip()
    metadata['participant_id'] = metadata['participant_id'].str.strip()
    metadata = apply_manual_fixes(metadata)

    metadata.insert(0, 'identifier', metadata.apply(lambda x: "{}_{}_{}_{}".format(x.participant_id, x.task_name, x.modality, x.video_name), axis=1))
    metadata.drop_duplicates('identifier', inplace=True)

    # Remove some videos 

    # i) Tobii-extracted videos
    metadata['is_recording_video'] = metadata.video_name.apply(lambda x: True if x.startswith('Recording') else False)
    metadata = metadata[~metadata['is_recording_video']]
    metadata.drop(columns='is_recording_video', inplace=True)
    pattern_1 = r' from \d{2}\.\d{2}\.\d{2} to \d{2}\.\d{2}\.\d{2}\.mp4$'
    pattern_2 = r' \d{2}\.\d{2}\.\d{2}\.mp4$'
    metadata['is_tobii_output'] = metadata.video_path.apply(lambda x: (bool(re.search(pattern_1, x) or bool(re.search(pattern_2, x))) ))
    metadata = metadata[~metadata['is_tobii_output']]
    metadata.drop(columns='is_tobii_output', inplace=True)

    # Append order for the Tobii video
    #metadata['order'] = metadata.groupby(['task_name', 'participant_id', 'modality'])['date'].transform(lambda x: x.argsort() if pd.notnull(x).all() else [np.nan]*len(x))

    # Cound partitions for each modality 
    metadata['n_videos'] = metadata.groupby(['task_name', 'participant_id', 'modality'])['identifier'].transform(lambda x: len(np.unique(x)))

    # Sort metadata and assign identifiers of passation, modality, and videos 
    metadata.sort_values(['task_name', 'participant_id', 'modality', 'video_name'], inplace=True)
    metadata = metadata.assign(passation_id = metadata.groupby(['task_name', 'participant_id']).ngroup())
    metadata = metadata.assign(modality_id = metadata.groupby(['task_name', 'participant_id', 'modality']).ngroup())
    metadata = metadata.assign(video_id = metadata.video_path.factorize()[0])

    # Apply final check, accounted for when collating video partitions
    metadata['is_checked'] = metadata.video_path.apply(check_video)

    # Save dataframe 
    metadata.to_csv(path_metadata, index=False)
    print('{} videos registered - total size: {} To'.format(metadata.shape[0], metadata.size.sum()/1024))


def parse_args():
    
    parser = argparse.ArgumentParser(description='Register Smartflat dataset')
    parser.add_argument('-o', '--output_path', default=None, help='Path to save the output (default to persistent_metadata folder)')
    parser.add_argument('-s', '--snapshot', action="store_true", default=False, help='After registration, whether or not creating frames snapshots of the datasets')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
            
    if args.output_path is None:
        path_metadata  = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', 'metadata.csv')

    main(path_metadata=path_metadata)
    print('Registration done.')
    
    if args.snapshot:
        
        main_snapshot(path_metadata=path_metadata)
        print("Snapshots created.")
        
    sys.exit(0)
    
    