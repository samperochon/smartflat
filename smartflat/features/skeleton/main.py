import argparse
import json
import logging
import os
import socket
import sys
import time

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, bridge, cpu, gpu
from tqdm import tqdm

bridge.set_bridge('native')

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import (
    fetch_output_path,
    get_api_root,
    get_data_root,
    get_host_name,
    get_video_loader,
    parse_flag,
    parse_identifier,
)

log_dir = os.path.join(get_data_root(), 'log');  os.makedirs(log_dir, exist_ok=True)
                    
logging.basicConfig(filename=os.path.join(log_dir,'skeleton_estimation_computation.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logging.info("Skeleton landmarks estimation computation")
logger = logging.getLogger('main')


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result['pose_landmarks']
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark['x'], y=landmark['y'], z=landmark['z']) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def create_skeleton_video(identifier, skeleton_landmarks_path, video_path, overwrite=False):
    
    
    # output generation preparation
    participant_id, task_name, modality, video_name = parse_identifier(identifier)

    os.makedirs(os.path.join(get_data_root(), 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(get_data_root(), 'outputs', 'skeleton_landmarks'), exist_ok=True)
    os.makedirs(os.path.join(get_data_root(), 'outputs', 'skeleton_landmarks', participant_id), exist_ok=True)
    os.makedirs(os.path.join(get_data_root(), 'outputs', 'skeleton_landmarks', participant_id, modality), exist_ok=True)

    ouptut_path = os.path.join(get_data_root(), 'outputs', 'skeleton_landmarks', participant_id, modality,  f'{identifier}_skeleton_landmarks_plot.mp4')
    if os.path.isfile(ouptut_path) and not overwrite:
        print("Video already exists in {} (overwrite={})".format(ouptut_path, overwrite))
        return None
    
    print(f"[Video creation] processing {video_path}...")

    with open(skeleton_landmarks_path, 'r') as f:
        data = json.load(f)
        
    vr = VideoReader(video_path, num_threads=5, ctx=cpu(0))

    print(f"Creating annotated video array ({len(data)} frames)")
        
    fps =  vr.get_avg_fps(); sampling_frequency = 1/2

        
    

    frames = [] 
    # Create the array of frames with annotations
    #for i, pose_landmarker_result in tqdm(enumerate(data)):
    for i, idx in enumerate(tqdm(range(0, len(vr), int(fps * sampling_frequency)))):
        
        pose_landmarker_result = data[i]
        
        rgb_image = vr[idx].numpy()
        initial_height, initial_width = rgb_image.shape[:2]
        new_size = (initial_width // 3, initial_height // 3)
        rgb_image = cv2.resize(rgb_image, new_size)
        
        annotated_image = draw_landmarks_on_image(rgb_image, pose_landmarker_result)
        frames.append(annotated_image)
    
    print('done')

    frames = np.array(frames)
    print("stats: ", frames.shape, frames[0].shape)
    
    # Obtenez les dimensions de la frame
    height, width, layers = frames[0].shape

    # Définissez le codec et créez un objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ou utilisez 'XVID'
    fps = int(vr.get_avg_fps()/2)
    video = cv2.VideoWriter(ouptut_path, fourcc, fps, (width, height))

    # Écrivez chaque frame dans le fichier vidéo
    for frame in frames:
        video.write(frame)

    # Libérez les ressources
    video.release()

    print("Skeleton video saved in {}".format(ouptut_path))
    print("rsync -ahuvz --progress  cheetah:/diskA/sam_data/data/outputs  /Users/samperochon/Borelli/algorithms/data/")
    return frames


def main():
        
    # -------- 
    # Define model and video dataset path
    path = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', '{}_dataset_df.csv'.format(get_host_name()))
    model_ckpt = os.path.join(get_api_root(), 'api', 'features', 'skeleton', 'model', 'pose_landmarker_heavy.task')
    model_name = 'skeleton_landmarks_mediapipe'
    os.makedirs(os.path.join(get_data_root(), 'dataframes', 'frozen-metrics-logs'), exist_ok=True)
    metrics_path = os.path.join(get_data_root(), 'dataframes', 'frozen-metrics-logs', f'{get_host_name()}_compute_time_skeleton_landmarks_detection.csv')
    sampling_frequency = 1/2 # Estimate skeleton every half a second (multiplied by the fps of the video in practice)
    max_num_poses = 3
    logging_interval = 2000
        

    # Setup mediapipe
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_ckpt),
        running_mode=VisionRunningMode.VIDEO, 
        output_segmentation_masks=True,
        num_poses=max_num_poses
    )

    
    # Get video to process
    dset = get_dataset(dataset_name='skeleton_landmarks', root_dir=get_data_root('light'), scenario='gold_unprocessed')
    #dset = get_dataset(dataset_name='skeleton_landmarks',  root_dir='/gpfs/workdir/perochons/data-gold-final', scenario='unprocessed')
    
    video_paths = dset.metadata['video_path'].tolist()
    
    metrics = [] 
    for j, video_path in enumerate(video_paths):

        start_time = time.time()
        logger.info("Extracting skeleton landmarks {}/{} for {}".format(j+1, len(video_paths), video_path))
        
        skeleton_landmark_output_path = fetch_output_path(video_path, model_name)
        if os.path.isfile(skeleton_landmark_output_path) and (parse_flag(skeleton_landmark_output_path, 'skeleton_landmarks_representation') == 'success'):
            #TODO: should be deprecated as this s handle throuhj 
            logger.info("Computation aready done. (should be deprecated as handled at the dataset level)")
            continue

        try:
            vr = VideoReader(video_path, num_threads=5, ctx=cpu(0))

        except:
            
            with open(os.path.join(os.path.dirname(skeleton_landmark_output_path), f'.{os.path.basename(video_path)[:-4]}_skeleton_landmarks_flag.txt'), 'w') as f:
                f.write('failure')
            logger.info("[FAILURE] Loading video for {}".format(skeleton_landmark_output_path))
            continue

        fps =  vr.get_avg_fps(); n_frames =  vr._num_frame; duration = n_frames/fps
        results = []
        with PoseLandmarker.create_from_options(options) as landmarker:
                
            for i, idx in enumerate(tqdm(range(0, len(vr), int(fps * sampling_frequency)))):
                
                try:
                    image = vr[idx].numpy()
                except:
                    with open(os.path.join(os.path.dirname(skeleton_landmark_output_path), f'.{os.path.basename(video_path)[:-4]}_skeleton_landmarks_flag.txt'), 'w') as f:
                        f.write('failure')
                    logger.info("[FAILURE] Error loading frame {} for {}".format(idx, skeleton_landmark_output_path))
                    raise ValueError

                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                frame_timestamp_ms = int(idx / n_frames * duration * 1e3)
                
                # Perform hand landmarks detection on the provided single image.
                pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                results.append(pose_landmarker_result)

                if i % logging_interval == 0:
                    logger.info("{}/{} {} - {}/{} [{:.2f} %]".format(j, len(video_paths), video_path, i, len(vr), 100*i/len(vr)))
                    
        # Success flag
        with open(os.path.join(os.path.dirname(skeleton_landmark_output_path), f'.{os.path.basename(video_path)[:-4]}_skeleton_landmarks_flag.txt'), 'w') as f:
            f.write('success')
        logger.info("Wrote success flags: {}".format(os.path.dirname(skeleton_landmark_output_path)))
        print('Extracted landmarks for {}'.format(skeleton_landmark_output_path))
        # Save results and metrics
        with open(skeleton_landmark_output_path, 'w') as f:
            json.dump(results, f, default=lambda x: x if type(x) == list else x.__dict__)
            
        metrics.append({'video_path': video_path, 'num_frames': len(vr), 'skeleton_landmarks_compute_time': time.time() - start_time, 'skeleton_sampling_frequency': sampling_frequency})
        if os.path.isfile(metrics_path):
            metricsdf = pd.concat([pd.read_csv(metrics_path), pd.DataFrame(metrics)])
        else:
            metricsdf =  pd.DataFrame(metrics)
        metricsdf.to_csv(metrics_path, index=False)

    sys.exit(0)

    


def get_args():
    parser = argparse.ArgumentParser(
        'Extract Skeleton landmarks using Mediapipe', add_help=False)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()    
    main()
