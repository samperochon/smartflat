"""Extract features for temporal action detection datasets


Sam Perochon, adapted from https://github.com/OpenGVLab/VideoMAEv2. December 2023.

"""
import argparse
import logging
import math
import os
import random
import socket
import sys
import time

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import numpy as np
import pandas as pd
import torch
from timm.models import create_model
from torchvision import transforms
from tqdm import tqdm

#import deepspeed

#TOFIX with eg export PYTHONPATH=/path/to/your/package:$PYTHONPATH

from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import (
    fetch_output_path,
    get_api_root,
    get_data_root,
    get_free_space,
    get_host_name,
    get_video_loader,
)

log_dir = os.path.join(get_data_root(), 'log');  os.makedirs(log_dir, exist_ok=True)
                    
logging.basicConfig(filename=os.path.join(log_dir,'video_representations_computation.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logging.info("Video Foundation Model Computation")
logger = logging.getLogger('main')

def main(device='0', chunk_idx=0, num_chunks=10, path_df_external=None, do_reversed=False):
    
    # Ensure chunk_idx is within the valid range
    assert 0 <= chunk_idx < num_chunks, "chunk_idx must be between 0 and num_chunks-1"
    
    print(f'chunk_idx: {chunk_idx} and num_chucks: {num_chunks}')

    if path_df_external is not None:
        df_external = pd.read_csv(path_df_external)
        
    # Bottleneck
    #free_space = get_free_space(get_data_root())
    #print('\nThe free space in {} is {:.2f} GB'.format(socket.gethostname(), free_space))

    # Define model and video dataset path #TOFIX: use config
    path = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', '{}_dataset_df.csv'.format(socket.gethostname()))
    model_ckpt = os.path.join(get_api_root(), 'api', 'features', 'video', 'models', 'vit_g_hybrid_pt_1200e_k710_ft.pth')
    model_name = 'vit_giant_patch14_224'
    

    os.makedirs(os.path.join(get_data_root(), 'dataframes', 'frozen-metrics-logs'), exist_ok=True)
    metrics_path = os.path.join(get_data_root(), 'dataframes', 'frozen-metrics-logs', f'{get_host_name()}_compute_time_video_representation_learning.csv')
    
    segment_length = 16 # Frame per segments
    delta_t = 8 # Elapsed number of frames between each temporal segments being represented
    logging_interval = 1000
    
    # Get video loader 
    video_loader = get_video_loader()
    start_idx_range = smartflat_range
    transform = transforms.Compose([ToFloatTensorInZeroOne(),Resize((224, 224))])

    # Get model
    model = get_model(model_name, model_ckpt)
    model.to(device)
    
    # Get video to process
    dset = get_dataset(dataset_name='video_block_representation', root_dir=get_data_root('light'), scenario='gold_unprocessed')
    #dset = get_dataset(dataset_name='video_block_representation', root_dir='/gpfs/workdir/perochons/data-gold-final', scenario='unprocessed')

    
    video_paths = dset.metadata['video_path'].tolist()

    # Split video_paths into chunks
    chunk_size = math.ceil(len(video_paths) / num_chunks)
    chunks = [video_paths[i:i + chunk_size] for i in range(0, len(video_paths), chunk_size)]

    # Get the specific chunk to process
    try:
        video_paths_chunk = chunks[chunk_idx]
    except:
        
        video_paths_chunk = video_paths
    if do_reversed:
        video_paths_chunk = list(reversed(video_paths_chunk))
        
    print('Processing files:')
    print('\n'.join(video_paths_chunk))
    
    
    metrics = [] 
    for idx, video_path in enumerate(video_paths_chunk):
        
        if video_path == '/gpfs/workdir/perochons/data-gold-final/lego/L154_P113_RAYVia_03052023/GoPro2/merged_video.mp4':
            print('WARNING passn /gpfs/workdir/perochons/data-gold-final/lego/L154_P113_RAYVia_03052023/GoPro2/merged_video.mp4')
            continue
        start_time = time.time()
        logger.info("Extracting feature {}/{} for {}".format(idx+1, len(video_paths_chunk), video_path))
        
        output_file = fetch_output_path(video_path, model_name)

        if os.path.isfile(output_file):
            logger.info("Already done.")
            continue
            
        try:
            vr = video_loader(video_path)
        except:            
            # failure flag
            print("[FAILURE] Unreadable video {}".format(video_path))
            logger.info("[FAILURE] Unreadable video {}".format(video_path))
            with open(os.path.join(os.path.dirname(output_file), f'.{os.path.basename(video_path)[:-4]}_video_representation_flag.txt'), 'w') as f:
                f.write('failure')
            continue

        if len(vr) <= segment_length:
            print("[FAILURE] Video duration does not meet minimum length {}".format(video_path))
            # failure flag
            logger.info("[FAILURE] Video duration does not meet minimum length {}".format(video_path))
            with open(os.path.join(os.path.dirname(output_file), f'.{os.path.basename(video_path)[:-4]}_video_representation_flag.txt'), 'w') as f:
                f.write('failure')
                
            continue
            
        feature_list = []
        for i, start_idx in enumerate(tqdm(start_idx_range(len(vr), segment_length, delta_t))):
            
            try:
                data = vr.get_batch(np.arange(start_idx, start_idx + segment_length))#.asnumpy()
                #frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
                frame_q = transform(data)  # torch.Size([3, 16, 224, 224])
                input_data = frame_q.unsqueeze(0).to(device)

                with torch.no_grad():
                    feature = model.forward_features(input_data)
                    feature_list.append(feature.cpu().numpy())
            except:
                feature_list.append(np.zeros((1, 1408)))
    

            if i % logging_interval == 0:
                logger.info("{}/{} {} - {}/{} [{:.2f} %]".format(idx, len(video_paths_chunk), video_path, i, len(start_idx_range(len(vr), segment_length, delta_t)), 100*i/len(start_idx_range(len(vr), segment_length, delta_t))))

        # Add check to see if all embeddings computation failed
        if np.vstack(feature_list).mean() == 0:    

            logger.info("[FAILURE] All block embeddings are null {}".format(video_path))
            with open(os.path.join(os.path.dirname(output_file), f'.{os.path.basename(video_path)[:-4]}_video_representation_flag.txt'), 'w') as f:
                f.write('failure')
                
        
        # [N, C]
        np.save(output_file, np.vstack(feature_list))
        metrics.append({'video_path': video_path, 'num_frames': len(vr), 'video_representations_compute_time': time.time() - start_time, 'segment_length': segment_length, 'delta_t': delta_t})
        logger.info(f'[{output_file} / {len(video_paths_chunk)}]: saved feature on {output_file}')
        logger.info("Elapsed time: {:.2f}s".format(time.time() - start_time))

        # Save success flag
        with open(os.path.join(os.path.dirname(output_file), f'.{os.path.basename(video_path)[:-4]}_video_representation_flag.txt'), 'w') as f:
            f.write('success')
        print('Extracted block embedding for {}'.format(video_path))
                            
        
        # Save logging df
        if os.path.isfile(metrics_path):
            metricsdf = pd.concat([pd.read_csv(metrics_path), pd.DataFrame(metrics)])
        else:
            metricsdf =  pd.DataFrame(metrics)
        metricsdf.to_csv(metrics_path, index=False)

    sys.exit(0)




def smartflat_range(num_frames, segment_length, delta_t):
    """For the smarflat project we want precise modeling of the video, 
    with baseline representations having overlap between each (so delta_t < segment_length)"""
    return range(0, num_frames - segment_length-1, delta_t)

def get_model(model_name, model_ckpt):
    # get model & load ckpt
    model = create_model(
        model_name,
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True)
    ckpt = torch.load(model_ckpt, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    
    if False:
        model = deepspeed.init_inference(model,
                                    tensor_parallel={"tp_size": 1},
                                    dtype=torch.float16,
                                    checkpoint=None,
                                    replace_with_kernel_inject=False
                                    ).module
        print('[INFO] Using Deepspeed and model Quantization to speed-up inference.')
    
    #print("Cutting model...")
    #del model.blocks[2:35]
    return model

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)



def parse_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--cuda',
        default='0',
        type=str,
        help='GPU id to use')

    parser.add_argument('--chunk_idx', type=int, default=0, help='Index of the chunk to process')
    parser.add_argument('--num_chunks', type=int, default=10, help='Number of chunks to split the workload into')
    parser.add_argument('-r', '--do_reversed', action="store_true", default=False, help='Reverse the list of video paths to process.')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    main(device, num_chunks=args.num_chunks, chunk_idx=args.chunk_idx, do_reversed=args.do_reversed)
