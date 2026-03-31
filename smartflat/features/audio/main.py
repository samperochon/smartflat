"""Extract features for temporal action detection datasets

Adapated from https://github.com/m-bain/whisperX and https://huggingface.co/intfloat/multilingual-e5-large. 

Sam Perochon, December 2023.

""" 
import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import (
    fetch_flag_path,
    fetch_output_path,
    get_api_root,
    get_data_root,
    get_host_name,
    get_video_loader,
    parse_flag,
    parse_identifier,
    print_nvidia_smi_output,
)

log_dir = os.path.join(get_data_root(), 'log');  os.makedirs(log_dir, exist_ok=True)
                    
logging.basicConfig(filename=os.path.join(log_dir,'audio_representations_computation.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logging.info("Speech Recognition and Text embedding computation")
logger = logging.getLogger('main')


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def parse_json(path):

    with open(path, 'r') as f:
        data = json.load(f)

    input_texts = [seg['text'] for seg in data['segments']]
    starts = [seg['start'] for seg in data['segments']]
    ends = [seg['end'] for seg in data['segments']]
    confidences = [[word['score'] if 'score' in word else np.nan   for word in seg['words']] for seg in data['segments'] ]
    speaker = [[word['speaker'] if 'speaker' in word else np.nan   for word in seg['words']] for seg in data['segments'] ]
    return starts, ends, input_texts, confidences, speaker

def compute_speech_embeddings(input_texts, tokenizer, model):
        
     # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.detach().numpy()

def main(whisperx_bin_file):
    # -------- 
    # Init. variables
    path = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', '{}_dataset_df.csv'.format(get_host_name()))
    speech_recognition_model_name = 'whisperx' #https://github.com/m-bain/whisperX/tree/main?tab=readme-ov-file 
    speech_embedding_model_name = 'multilingual-e5-large' 
    logging_interval = 2000
    
    print('Entering main of speech feature extraction...')
    print_nvidia_smi_output()
    


    # Get model ready
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    
    base_command = [whisperx_bin_file,
                    '--device_index', '0',
                    '--model', 'large', 
                    '--language', 'fr', 
                    '--diarize', 
                    '--highlight_words', 'True',
                    '--min_speakers', '0', 
                    '--max_speakers', '3', 
                    '--output_format', 'all', #TODO: set to 'json'
                    '--hf_token', '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']
    
    # Get video to process
    dset = get_dataset(dataset_name='speech_recognition_representation', root_dir=get_data_root('light'), scenario='gold_unprocessed')
    #dset = get_dataset(dataset_name='speech_recognition_representation', root_dir='/gpfs/workdir/perochons/data-gold-final', scenario='unprocessed')
     
    video_paths = dset.metadata['video_path'].tolist()
    
    print(f'Unprocessed:')
    print('\t\t\t\t\n'.join(video_paths))
      
    # Process..
    for i, video_path in enumerate(reversed(video_paths)):
        
        # if i < 7 : 
        #     print('Skip {}'.format(video_path))
        #     continue

        logger.info("Extracting speech {}/{} for {}".format(i+1, len(video_paths), video_path))
        print("Extracting speech {}/{} for {}".format(i+1, len(video_paths), video_path))
        
        speech_recognition_output_path = fetch_output_path(video_path, speech_recognition_model_name)
        speech_representation_output_path = fetch_output_path(video_path, speech_embedding_model_name)
        speech_recognition_flag_path = fetch_flag_path(speech_recognition_output_path, 'flag_speech_recognition')
        speech_representation_flag_path = fetch_flag_path(speech_representation_output_path, 'flag_speech_representation')

        # Ajoutez le chemin du fichier et le répertoire de sortie à la commande
        command = base_command + [video_path, '--output_dir', os.path.dirname(video_path)]

        #try:
        subprocess.run(command, check=True)

        speech_recognition_raw_output_path = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + '.json')
        
        # Rename output path to our conventions
        assert os.path.isfile(speech_recognition_raw_output_path)
        subprocess.run(['mv', speech_recognition_raw_output_path, speech_recognition_output_path])
        assert os.path.isfile(speech_recognition_output_path)

        # Success flag
        with open(speech_recognition_flag_path, 'w') as f:
            f.write('success')
        logger.info("Speech recognition wrote success flags.")
        print("Speech recognition wrote success flags.")

        #except:            
        # Failure flag
        # TOFIX: for now, since the error may due to ressources limitation running the command, and because the computation is relatively light, we don't use failure flags. 
        #with open(os.path.join(os.path.dirname(os.path.dirname(speech_recognition_output_path)), f'.{os.path.basename(video_path)[:-4]}_speech_recognition_flag.txt'), 'w') as f:
        #    f.write('failure')
        #logger.info("[FAILURE] Error extracting speech recognition for {}".format(video_path))
        

        
        # From here we assume the speech recognition algorithms successfully created the followingoutput file with detected text, confidence scores, and spekers estimation ids. 
        starts, ends, input_texts, confidences, speaker = parse_json(speech_recognition_output_path)

        try:
            embeddings = compute_speech_embeddings(input_texts, tokenizer, model)

            # Success flag
            with open(speech_representation_flag_path, 'w') as f:
                f.write('success')
                logger.info("Wrote success flags: {}".format(speech_representation_flag_path))
        except:
            with open(speech_representation_flag_path, 'w') as f:
                f.write('failure')
            logger.info("[FAILURE] Error extracting speech representation for {}".format(video_path))
            continue
        
        # [N, D]
        np.save(speech_representation_output_path, embeddings)

        logger.info("{}/{} done. {}".format(i+1, len(video_paths), video_path))
        
        print("{}/{} done. {}".format(i+1, len(video_paths), video_path))
        
        print('Extracted audio for {}'.format(video_path))

def get_args():
    parser = argparse.ArgumentParser(
        'Extract speech recognition and representation using WhisperX and Multilingual model', add_help=False)
    return parser.parse_args()


if __name__ == '__main__':
    
    host_name = get_host_name()
    print('Host name: {}'.format(host_name))
    if host_name == 'cheetah':
        whisperx_bin_file = '/home/sam/miniconda3/envs/whisperx/bin/whisperx'
    elif 'ruche' in host_name:
        whisperx_bin_file = '/gpfs/workdir/perochons/.conda/envs/whisperx/bin/whisperx'
    else:
        raise ValueError("Unknown host for speech computation: {}".format(host_name))
    
    args =get_args()   
    main(whisperx_bin_file=whisperx_bin_file)
    sys.exit(0)

