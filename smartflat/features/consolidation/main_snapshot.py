"""Plot the images of the present videos in the data/snapshots/ folder."""
import argparse
import os
import socket
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decord import VideoReader, bridge, cpu

bridge.set_bridge('native')


from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import check_video, get_data_root, get_file_size_in_gb, parse_task_number, parse_participant_id

OPEN_CV_BACKEND = False 
L, l, c = 224, 398, 3

def main(output_folder = None, media_type='video_frame', overwrite=False):
    '''From a metadata file keyed by video, plot images with partitipants as rows and modality as columns in a structured fashion.
    
        For each partitipant (rows) and modality (e.g. GoPro 1,2,3 or Tobii), we select the longest video of the folder and plot the middle frame.
        If the video is not found, a black square is plotted.
        
        Screenshots images are identified by their unique participant_id folder name (p_id), modality_id (m_id) and video_id (v_id)
        to simplify visual inspection characterization of the dataset.
        
        Screenshots are saved in the get_data_root()/screenshots folder.
        Provided metadata must have valid `video_path` column.
    '''
    
    cols = ['video_path', 'task_name', 'participant_id', 'modality', 
       'video_name', 'identifier', 'has_video', 
       'n_videos', 'video_name_list', 'dates_list', 'fps_percy', 'n_fps',
       'size_list', 'size_percy', 
       'duration_list', 'duration_percy','duration', 'fps',
       'n_frames', 'date', 'size', 'is_checked','group_tmp',
       'trigram',  'folder_modality',
        'passation_id', 'modality_id', 
       'is_fish_eyed', 'upside_down', 'is_swapped_modality',
       'true_modality', 'GP3_is_sink', 'GP2_is_wrong_buttress',
       'GP2_above', 'GoPro1_is_wrong_buttress', 'is_middle_range',
       'true_task_name', 'is_old_setup', 'is_old_recipe', 'annot_notes',
       ]
    
    
    if media_type == 'video_frame':
        print('Extracting middle frames of videos')
        dset = get_dataset(dataset_name="base", scenario="all")
    elif media_type == 'gram':
        print('Extracting gram matrices of videos')
        dset = get_dataset(dataset_name='video_block_representation', root_dir=get_data_root(), scenario='success')


    df = dset.metadata.copy()
    
    # elif path_metadata is None:
        
    #     path_metadata  = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', 'metadata.csv')
    #     print(f'Using dataset {dset} saved in  {path_metadata}')
    #     df = pd.read_csv(path_metadata)

    if output_folder is None:
        output_folder = os.path.join(get_data_root(), f'thumbnails-{media_type}-12112024')
            
    os.makedirs(output_folder, exist_ok=True)
    
    
    #print('Dataset description path:', path_metadata)
    print('Output thumbnail path:', output_folder)
    
    # Sort metadata and assign identifiers of passation, modality, and videos 
    #df.sort_values(['task_name', 'participant_id', 'modality', 'video_name'], inplace=True)
    
    # Deprecated ?
    #df = df.assign(passation_id = df.groupby(['task_name', 'participant_id']).ngroup())
    #df = df.assign(modality_id = df.groupby(['task_name', 'participant_id', 'modality']).ngroup())
    #df = df.assign(video_id = df.video_path.factorize()[0])
    
    

    print(f'Initial shape of metadata: {dset.metadata.shape}')
    df[["task_number", "diag_number", "trigram", "date_folder"]] = df["participant_id"].apply(parse_participant_id)
    df["task_number_int"] = df.task_number.apply(parse_task_number)
    df_middle_frames = (df.drop_duplicates(['task_name', 'participant_id', 'modality', 'video_name'], keep='first')
                                .dropna(subset=['video_path'] if media_type == 'video_frame' else [])
                                .sort_values(['task_name', 'task_number_int', 'participant_id', 'modality', ], ascending=True)
                                ).copy()
        
    # Créer une nouvelle colonne pour les groupes de 25
    df_middle_frames['passation_id'] = df.groupby(['task_name', 'participant_id']).ngroup()
    df_middle_frames['modality_id'] = df.groupby(['task_name', 'participant_id', 'modality']).ngroup()
    df_middle_frames['group_tmp'] = df_middle_frames.passation_id.factorize()[0] // 5
    
    # Save dataframes
    df.to_csv(os.path.join(output_folder, 'df.csv'), index=False)
    df_middle_frames[cols].to_csv(os.path.join(output_folder, 'df_middle_frames.csv'), index=False)
    print('Visual inspection dataframe saved in: {}'.format(os.path.join(output_folder, 'df_middle_frames.csv')))
    n_expected = len(df_middle_frames)
    print('Number of expected images:', n_expected)
    n = 0
    for group_tmp, group in df_middle_frames.groupby('group_tmp'):
        
        
        output_path = os.path.join(output_folder, f'middle_frame_{int(group.passation_id.min())}_{int(group.passation_id.max())}.png')
        if os.path.isfile(output_path) and not overwrite:
            print(f'Passation ids {int(group.passation_id.min())} to {int(group.passation_id.max())} already done.')
            continue
        
        fig, axs = plt.subplots(5, 4, figsize=(35, 30))
        fig.set_facecolor('white')


        for i, (passation_id, passation) in enumerate(group.groupby('passation_id')):

            participant_id = passation.participant_id.iloc[0]
            print(f'Extracting middle frame for participant {participant_id}')

            for j, modality in enumerate(['GoPro1', 'GoPro2', 'GoPro3', 'Tobii']):
                
                print(f'\t{modality}')

                to_plot = passation[passation['modality'] == modality]
                
                # if participant_id == 'G96_P82_CABAnt_22062022' and modality in ['GoPro2']:
                #     is_corrupted = True
                    
                # else:
                #     is_corrupted = False
                is_corrupted = False


                if len(to_plot) == 0 or is_corrupted:

                    frame = np.ones((L, l, c), dtype=np.uint8)
                    video_id = -1
                    modality_id = -1

                elif len(to_plot) == 1:
                    
                    row = to_plot.iloc[0]
                    
                    
                    if media_type == 'video_frame':

                        frame = extract_middle_frame(row.video_path)
                        frame = cv2.resize(frame, (l, L))  # resize to the target size
                        axs[i][j].imshow(frame)
                    
                    elif media_type == 'gram':
                        
                        frame = compute_gram(row)
                        axs[i][j].imshow(frame)
                        
                    
                    if frame.sum() == 267456:
                        print(f'Corrupted video ? Black image as middle frame: {row.video_path}')
                        video_id = row.video_id
                        modality_id = row.modality_id
                    else:
                        video_id = row.video_id
                        modality_id = row.modality_id
                            

                else:
                    raise ValueError

                
                axs[i][j].set_title(f"{participant_id}\np: {int(passation_id)} m: {int(modality_id)}", weight='bold', fontsize=18)
                axs[i][j].axis('off')  # pour masquer les axes

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f'Done passation {group.passation_id.unique()}')
    print('Done')
    return 


def extract_middle_frame(video_path):
    
    if OPEN_CV_BACKEND:
        cap = cv2.VideoCapture(video_path)

        # Obtenir le nombre total de frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Définir la frame à lire comme étant la frame du milieu
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

        # Lire la frame
        ret, frame = cap.read()

        # Si la frame a été lue correctement, alors ret est True
        if ret:
            # Convertir la frame en RGB pour l'affichage avec matplotlib
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.ones((L, l, c), dtype=np.uint8)

        # Libérer les ressources
        cap.release()
    else:
        try:
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        except:
            print(f'Error reading {video_path}')
            import subprocess
            #subprocess.run(['open', os.path.dirname(video_path)])
            subprocess.run(['rm', video_path])
            #print('Opening folder: ',  os.path.dirname(video_path))
            return np.ones((L, l, c), dtype=np.uint8)
        
        total_frames =  vr._num_frame
        frame = vr[total_frames // 2].asnumpy()
        
    return frame

def compute_gram(row):
    
    x = np.load(row.video_representation_path)
    gram = x @ x.T

    return gram 

def parse_args():
    
    parser = argparse.ArgumentParser(description='Collate dataset')
    parser.add_argument('-p', '--path_metadata', default='metadata', help='local or metadata')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
            
    if args.path_metadata == 'local':
        
        root_dir = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', '{}_dataset_df.csv'.format(socket.gethostname()))
        
        
    elif args.path_metadata == 'metadata':
        
        path_metadata  = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', 'metadata.csv')
        
    output_folder = '/home/perochon/data-gold-final/thumbnails/thumbnails-28022024'
    main(output_folder=output_folder)
    sys.exit(0)    