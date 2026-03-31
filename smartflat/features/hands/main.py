import argparse
import json
import logging
import os
import socket
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from decord import DECORDError, VideoReader, bridge, cpu, gpu
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from tqdm import tqdm

bridge.set_bridge('torch')


from smartflat.datasets.loader import get_dataset  # noqa: E402
from smartflat.utils.utils_io import (  # noqa: E402
    fetch_output_path,
    get_api_root,
    get_host_name,
    get_data_root,
    get_video_loader,
    parse_flag,
    parse_identifier,
)

log_dir = os.path.join(get_data_root(), 'log');  os.makedirs(log_dir, exist_ok=True)
                    
logging.basicConfig(filename=os.path.join(log_dir, 'hand_landmarks_representations_computation.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logging.info("Hands landmarks estimation computation")
logger = logging.getLogger('main')

# Visualisation

def parse_json(hand_landmarks_path):
    """Parse the hand landmarks detection output file. 
    
    Notes:
        - Create two dataframes for each hands, gathering data on hand detection during all frames of the video
        - 
        """

    with open(hand_landmarks_path, 'r') as f:
        data = json.load(f)

    res_l = []
    res_r = []
    for i, Hi in enumerate(data):
        if len(Hi['handedness']) == 0:
            res_l.append({'landmarks': np.array([3 * [np.nan] for _ in range(21)])[None, :, :],
                                   'w_landmarks': np.array([ 3 * [np.nan]  for _ in range(21)])[None, :, :],
                                   'visibility': np.array([np.nan  for _ in range(21)])[None, :],
                                   'presence': np.array([np.nan  for _ in range(21)])[None, :],
                                   'detected': False,
                                  })
            res_r.append({'landmarks': np.array([3 * [np.nan] for _ in range(21)])[None, :, :],
                                   'w_landmarks': np.array([ 3 * [np.nan]  for _ in range(21)])[None, :, :],
                                   'visibility': np.array([np.nan  for _ in range(21)])[None, :],
                                   'presence': np.array([np.nan  for _ in range(21)])[None, :],
                                   'detected': False,
                                  })        
            
        elif len(Hi['handedness']) == 1:
            
            handedness, hand_landmarks, hand_world_landmarks = Hi['handedness'][0], Hi['hand_landmarks'][0], Hi['hand_world_landmarks'][0]
            
            if handedness[0]['category_name'] == 'Left':
                res_l.append({'landmarks': np.array([[d['x'], d['y'], d['z']] for d in hand_landmarks])[None, :, :],
                               'w_landmarks': np.array([[d['x'], d['y'], d['z']] for d in hand_world_landmarks])[None, :, :],
                               'visibility': np.array([d['visibility'] for d in hand_landmarks])[None, :],
                               'presence': np.array([d['presence'] for d in hand_landmarks])[None, :],
                               'detected': True,
                              })
                res_r.append({'landmarks': np.array([3 * [np.nan] for _ in range(21)])[None, :, :],
                                       'w_landmarks': np.array([ 3 * [np.nan]  for _ in range(21)])[None, :, :],
                                       'visibility': np.array([np.nan  for _ in range(21)])[None, :],
                                       'presence': np.array([np.nan  for _ in range(21)])[None, :],
                               'detected': False,
                                      })   
                    
                    
            elif handedness[0]['category_name'] == 'Right':
                res_l.append({'landmarks': np.array([3 * [np.nan] for _ in range(21)])[None, :, :],
                                       'w_landmarks': np.array([ 3 * [np.nan]  for _ in range(21)])[None, :, :],
                                       'visibility': np.array([np.nan  for _ in range(21)])[None, :],
                                       'presence': np.array([np.nan  for _ in range(21)])[None, :],
                               'detected': False,
                                      })
                
                res_r.append({'landmarks': np.array([[d['x'], d['y'], d['z']] for d in hand_landmarks])[None, :, :],
                               'w_landmarks': np.array([[d['x'], d['y'], d['z']] for d in hand_world_landmarks])[None, :, :],
                               'visibility': np.array([d['visibility'] for d in hand_landmarks])[None, :],
                               'presence': np.array([d['presence'] for d in hand_landmarks])[None, :],
                               'detected': True,
                              })
        
        else:
            # Collect handedness
            for handedness, hand_landmarks, hand_world_landmarks  in zip(Hi['handedness'], Hi['hand_landmarks'], Hi['hand_world_landmarks']):
                
                
                assert handedness[0]['display_name'] == handedness[0]['category_name']
                
                if len(handedness) > 1:
                    raise ValueError
                    
    
                if handedness[0]['category_name'] == 'Left':
                     res_l.append({'landmarks': np.array([[d['x'], d['y'], d['z']] for d in hand_landmarks])[None, :, :],
                                   'w_landmarks': np.array([[d['x'], d['y'], d['z']] for d in hand_world_landmarks])[None, :, :],
                                   'visibility': np.array([d['visibility'] for d in hand_landmarks])[None, :],
                                   'presence': np.array([d['presence'] for d in hand_landmarks])[None, :],
                                   'detected': True,
                                  })
                elif handedness[0]['category_name'] == 'Right':
                    res_r.append({'landmarks': np.array([[d['x'], d['y'], d['z']] for d in hand_landmarks])[None, :, :],
                                   'w_landmarks': np.array([[d['x'], d['y'], d['z']] for d in hand_world_landmarks])[None, :, :],
                                   'visibility': np.array([d['visibility'] for d in hand_landmarks])[None, :],
                                   'presence': np.array([d['presence'] for d in hand_landmarks])[None, :],
                                   'detected': True,
                                  })
                else:
                    raise ValueError 
    
    res_l = pd.DataFrame(res_l)
    res_r = pd.DataFrame(res_r)
    assert np.vstack(res_r[res_r['detected']]['presence']).sum() == 0
    assert np.vstack(res_l[res_l['detected']]['presence']).sum() == 0
    assert np.vstack(res_r[res_r['detected']]['visibility']).sum() == 0
    assert np.vstack(res_l[res_l['detected']]['visibility']).sum() == 0

    return res_l, res_r

# def create_hand_video(identifier, hand_landmarks_path, video_path, downsampling_factor=1000, overwrite=False):
#     """Create a video with hand landmarks annotations."""
    
    
#     MARGIN = 10  # pixels
#     FONT_SIZE = 1
#     FONT_THICKNESS = 1
#     HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    
#     # output generation preparation
#     participant_id, task_name, modality, video_name = parse_identifier(identifier)

#     os.makedirs(os.path.join(get_data_root(), 'outputs'), exist_ok=True)
#     os.makedirs(os.path.join(get_data_root(), 'outputs', 'hand_landmarks'), exist_ok=True)
#     os.makedirs(os.path.join(get_data_root(), 'outputs', 'hand_landmarks', participant_id), exist_ok=True)
#     os.makedirs(os.path.join(get_data_root(), 'outputs', 'hand_landmarks', participant_id, modality), exist_ok=True)

#     ouptut_path = os.path.join(get_data_root(), 'outputs', 'hand_landmarks', participant_id, modality,  f'{identifier}_hand_landmarks_plot.mp4')

#     if os.path.isfile(ouptut_path) and not overwrite:
#         print("Video already exists in {} (overwrite={})".format(ouptut_path, overwrite))
#         return None
    

#     print(f"[Video creation] processing {video_path}...")
    
    
#     with open(hand_landmarks_path, 'r') as f:
#         data = json.load(f)
        
#     vr = VideoReader(video_path, num_threads=5, ctx=cpu(0))
    
#     print(f"Creating annotated video array ({len(data)} frames)")

#     frames = []     
    
#     # Create the array of frames with annotations
#     for i, Hi in tqdm(enumerate(data)):
        
#         if i % downsampling_factor != 0:
#             continue
    
#         if len(Hi['handedness']) > 0:
            
#             try:
#                 rgb_image = vr[i].numpy()
#             except DECORDError:
#                 print(f'Error while loading frame {i} from {video_path}')
#                 rgb_image = np.zeros((100, 100, 3))
#                 rgb_image = np.zeros((360, 640, 3))

#             initial_height, initial_width = rgb_image.shape[:2]
#             # Calculer la nouvelle résolution tout en conservant le ratio
#             new_size = (initial_width // 3, initial_height // 3)

#             # Réduire la taille de la frame
#             rgb_image = cv2.resize(rgb_image, new_size)
    
#             hand_landmarks_list = Hi['hand_landmarks']
#             handedness_list = Hi['handedness']
#             annotated_image = np.copy(rgb_image)
    
#             # Loop through the detected hands to visualize.
#             for idx in range(len(hand_landmarks_list)):
#                 hand_landmarks = hand_landmarks_list[idx]
#                 handedness = handedness_list[idx]
    
#                 # Draw the hand landmarks.
#                 hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#                 hand_landmarks_proto.landmark.extend([
#                   landmark_pb2.NormalizedLandmark(x=landmark['x'], y=landmark['y'], z=landmark['z']) for landmark in hand_landmarks
#                 ])
#                 solutions.drawing_utils.draw_landmarks(
#                   annotated_image,
#                   hand_landmarks_proto,
#                   solutions.hands.HAND_CONNECTIONS,
#                   solutions.drawing_styles.get_default_hand_landmarks_style(),
#                   solutions.drawing_styles.get_default_hand_connections_style())
    
#                 # Get the top left corner of the detected hand's bounding box.
#                 height, width, _ = annotated_image.shape
#                 x_coordinates = [landmark['x'] for landmark in hand_landmarks]
#                 y_coordinates = [landmark['y'] for landmark in hand_landmarks]
#                 text_x = int(min(x_coordinates) * width)
#                 text_y = int(min(y_coordinates) * height) - MARGIN
    
#                 # Draw handedness (left or right hand) on the image.
#                 cv2.putText(annotated_image, f"{handedness[0]['category_name']}",
#                             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    
#             # STEP 5: Process the classification result. In this case, visualize it.
#             #annotated_image = draw_landmarks_on_image(image, results[-1])
#             frames.append(annotated_image)
    
#         else: # No hands detected
#             try:
#                 annotated_image = vr[i].numpy()
#             except DECORDError:
#                 print(f'Error while loading frame {i} from {video_path}')
#                 annotated_image = np.zeros((360, 640, 3))
                
            
#             initial_height, initial_width = annotated_image.shape[:2]
#             # Calculer la nouvelle résolution tout en conservant le ratio
#             size = (initial_width // 3, initial_height // 3)

#             # Réduire la taille de la frame
#             annotated_image = cv2.resize(annotated_image, size)
            
#             frames.append(annotated_image)

    
#     print('done')
    
#     frames = np.array(frames)
#     print("stats: ", frames.shape, frames[0].shape)
    
#     # Obtenez les dimensions de la frame
#     height, width, layers = frames[0].shape
    
#     # Définissez le codec et créez un objet VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ou utilisez 'XVID'
#     fps = vr.get_avg_fps()
#     video = cv2.VideoWriter(ouptut_path, fourcc, fps, (width, height))
    
#     # Écrivez chaque frame dans le fichier vidéo
#     for frame in frames:
#         video.write(frame)
    
#     # Libérez les ressources
#     video.release()
    
#     print("Hand video saved in {}".format(ouptut_path))
#     return frames

def create_hand_video(identifier, hand_landmarks_path, video_path, downsampling_factor=1000, overwrite=False, start_frame=0, end_frame=None):
    """Create a video with hand landmarks annotations for a specified range of frames."""
    
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    
    # output generation preparation
    participant_id, task_name, modality, video_name = parse_identifier(identifier)

    os.makedirs(os.path.join(get_data_root(), 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(get_data_root(), 'outputs', 'hand_landmarks'), exist_ok=True)
    os.makedirs(os.path.join(get_data_root(), 'outputs', 'hand_landmarks', participant_id), exist_ok=True)
    os.makedirs(os.path.join(get_data_root(), 'outputs', 'hand_landmarks', participant_id, modality), exist_ok=True)

    ouptut_path = os.path.join(get_data_root(), 'outputs', 'hand_landmarks', participant_id, modality,  f'{identifier}_hand_landmarks_plot.mp4')

    if os.path.isfile(ouptut_path) and not overwrite:
        print("Video already exists in {} (overwrite={})".format(ouptut_path, overwrite))
        return None

    print(f"[Video creation] processing {video_path}...")

    with open(hand_landmarks_path, 'r') as f:
        data = json.load(f)
        
    vr = VideoReader(video_path, num_threads=5, ctx=cpu(0))
    
    total_frames = len(data)
    if end_frame is None:
        end_frame = total_frames  # Process till the end if not specified

    print(f"Creating annotated video array from frames {start_frame} to {end_frame}")

    frames = []     
    processed_frame_count = 0  # Keep track of how many frames we've processed
    
    # Create the array of frames with annotations
    for i, Hi in tqdm(enumerate(data[start_frame:end_frame], start=start_frame)):
        
        if i % downsampling_factor != 0:
            continue
    
        if len(Hi['handedness']) > 0:
            try:
                rgb_image = vr[i].numpy()
            except DECORDError:
                print(f'Error while loading frame {i} from {video_path}')
                rgb_image = np.zeros((100, 100, 3))
                rgb_image = np.zeros((360, 640, 3))

            initial_height, initial_width = rgb_image.shape[:2]
            new_size = (initial_width // 3, initial_height // 3)
            rgb_image = cv2.resize(rgb_image, new_size)
    
            hand_landmarks_list = Hi['hand_landmarks']
            handedness_list = Hi['handedness']
            annotated_image = np.copy(rgb_image)
    
            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]
    
                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                  landmark_pb2.NormalizedLandmark(x=landmark['x'], y=landmark['y'], z=landmark['z']) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                  annotated_image,
                  hand_landmarks_proto,
                  solutions.hands.HAND_CONNECTIONS,
                  solutions.drawing_styles.get_default_hand_landmarks_style(),
                  solutions.drawing_styles.get_default_hand_connections_style())
    
                height, width, _ = annotated_image.shape
                x_coordinates = [landmark['x'] for landmark in hand_landmarks]
                y_coordinates = [landmark['y'] for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN
    
                cv2.putText(annotated_image, f"{handedness[0]['category_name']}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
            frames.append(annotated_image)
    
        else:  # No hands detected
            try:
                annotated_image = vr[i].numpy()
            except DECORDError:
                print(f'Error while loading frame {i} from {video_path}')
                annotated_image = np.zeros((360, 640, 3))
                
            initial_height, initial_width = annotated_image.shape[:2]
            size = (initial_width // 3, initial_height // 3)
            annotated_image = cv2.resize(annotated_image, size)
            frames.append(annotated_image)
        
        processed_frame_count += 1  # Increment the processed frames count
    
    print('done')
    
    frames = np.array(frames)
    print("stats: ", frames.shape, frames[0].shape)
    
    # Obtain the dimensions of the frame
    height, width, layers = frames[0].shape
    
    # Define codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    fps = vr.get_avg_fps()
    video = cv2.VideoWriter(ouptut_path, fourcc, fps, (width, height))
    
    # Write each frame to the video file
    for frame in frames:
        video.write(frame)
    
    # Release resources
    video.release()
    
    print(f"Hand video saved in {ouptut_path}")
    return frames 

def main(root_dir=None):
    
    if root_dir is None:
        root_dir = get_data_root('light')
        
                
    # -------- 
    # Define model and video dataset path
    path = os.path.join(root_dir, 'dataframes', 'persistent_metadata', '{}_dataset_df.csv'.format(get_host_name()))
    model_name = 'hand_landmarks_mediapipe'
    model_ckpt = os.path.join(get_api_root(), 'api', 'features', 'hands', 'model', 'hand_landmarker.task')
    os.makedirs(os.path.join(root_dir, 'dataframes', 'frozen-metrics-logs'), exist_ok=True)

    metrics_path = os.path.join(root_dir, 'dataframes', 'frozen-metrics-logs', f'{get_host_name()}_compute_time_hand_landmarks_detection.csv')
    max_num_hands = 4
    logging_interval = 2000


    # Setup mediapipe
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_ckpt),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=max_num_hands
        )


    # Get video to process
    dset = get_dataset(dataset_name='hand_landmarks', root_dir=root_dir, scenario='gold_unprocessed')
    #dset = get_dataset(dataset_name='hand_landmarks', root_dir='/gpfs/workdir/perochons/data-gold-final', scenario='unprocessed')

    
    #TODO Could be nice to use gethosname to automatize related states
    #extdf = pd.concat([pd.read_csv(os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', f'{machine_name}_dataset_df.csv')) for machine_name in ['ruche', 'cheetah']])
    #identifier_in_external_machine = extdf[~extdf['video_path'].isna()].identifier.to_list()
    #n = len(dset)
    #dset.metadata = dset.metadata[~dset.metadata.identifier.isin(identifier_in_external_machine)]
    #print(f'Filtered {n - len(dset)} videos from the dataset as found in remote.')
    print(f'Filtering with other remote DISABLED.')

    video_paths = dset.metadata.sort_values('size', ascending=True)['video_path'].tolist()
    
    print('Processing:')
    print(' '.join(video_paths))
    
    # Process..
    metrics = [] 
    for j, video_path in enumerate(video_paths):
        
        start_time = time.time()
        logger.info("Extracting hand landmarks {}/{} for {}".format(j+1, len(video_paths), video_path))
        
        hand_landmark_output_path = fetch_output_path(video_path, model_name)
        if os.path.isfile(hand_landmark_output_path):
            logger.info("Computation aready done. SHOULD BE DEPRECATED")
            #Sucess flag
            with open(os.path.join(os.path.dirname(hand_landmark_output_path), '.', f'.{os.path.basename(video_path)[:-4]}_hand_landmarks_flag.txt'), 'w') as f:
                f.write('success')
            continue

        try:
            vr = VideoReader(video_path, num_threads=5, ctx=cpu(0))
        except:
           #Failure flag
           with open(os.path.join(os.path.dirname(hand_landmark_output_path), '.', f'.{os.path.basename(video_path)[:-4]}_hand_landmarks_flag.txt'), 'w') as f:
               f.write('failure')
           logger.info("[FAILURE] Loading video for {}".format(hand_landmark_output_path))
           continue
            
        fps =  vr.get_avg_fps(); n_frames =  vr._num_frame; duration = n_frames/fps
        
        results = []
        n_failed = 0
        with HandLandmarker.create_from_options(options) as landmarker:
                
            for i, idx in enumerate(tqdm(range(len(vr)))):
        
                try:
                    image = vr[idx].numpy()
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                    frame_timestamp_ms = int(idx / n_frames * duration * 1e3)
                    
                    # Perform hand landmarks detection on the provided single image.
                    hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                    
                except:
                    hand_landmarker_result = mp.tasks.vision.HandLandmarkerResult(handedness=[], hand_landmarks=[], hand_world_landmarks=[]) 
                    n_failed += 1 
                results.append(hand_landmarker_result)

                if i % logging_interval == 0:
                    logger.info("{}/{} {} - {}/{} [{:.2f} %]".format(j, len(video_paths), video_path, i, len(vr), 100*i/len(vr)))


        if n_failed > 0.9 * len(vr):
            logger.info("[FAILURE] More than 90% of frames failed to be loaded.")
            with open(os.path.join(os.path.dirname(hand_landmark_output_path), '.', f'.{os.path.basename(video_path)[:-4]}_hand_landmarks_flag.txt'), 'w') as f:
                f.write('failure')
            continue
        
        #Sucess flag
        with open(os.path.join(os.path.dirname(hand_landmark_output_path), '.', f'.{os.path.basename(video_path)[:-4]}_hand_landmarks_flag.txt'), 'w') as f:
            f.write('success')
        logger.info("Wrote success flags: {}".format(os.path.dirname(hand_landmark_output_path)))

        print('Extracted landmarks for {}'.format(hand_landmark_output_path))

        # Save results and metrics
        with open(hand_landmark_output_path, 'w') as f:
            json.dump(results, f, default=lambda x: x if type(x) == list else x.__dict__)
            
        metrics.append({'video_path': video_path, 'num_frames': len(vr), 'hand_landmarks_compute_time': time.time() - start_time})
        if os.path.isfile(metrics_path):
            metricsdf = pd.concat([pd.read_csv(metrics_path), pd.DataFrame(metrics)])
        else:
            metricsdf =  pd.DataFrame(metrics)
        metricsdf.to_csv(metrics_path, index=False)
    

def get_args():
    parser = argparse.ArgumentParser(
        'Extract hands landmarks using Mediapipe', add_help=False)
    parser.add_argument('-r', '--root_dir', default='local', help='local or harddrive')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()    
    
    if args.root_dir == 'local':
        
        root_dir = None
        
    elif args.root_dir == 'harddrive':
        
        root_dir = '/Volumes/Smartflat/data' 
        
    print(root_dir)
    main(root_dir=root_dir)

    sys.exit(0)