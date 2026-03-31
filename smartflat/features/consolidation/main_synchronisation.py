"""Compute cross-modality audio synchronization via cross-correlation.

When to run: After audio extraction, to align GoPro and Tobii recordings.
Prerequisites: Dataset with extracted audio files across modalities.
Outputs: CSV with per-participant cross-correlation lags and confidence scores.
Usage: python -m smartflat.features.consolidation.main_synchronisation
"""

import argparse
import os
import sys
from itertools import combinations

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


from smartflat.constants import mapping_incorrect_modality_name, video_extensions
from smartflat.datasets.loader import get_dataset
from smartflat.features.consolidation.main_snapshot import main as main_snapshot
from smartflat.utils.utils_io import extract_audio, get_data_root


def main(root_dir=None, verbose=False):
    """From a data directory, compute all possible modality-level synchronisation, run checks, and save results.
    
    TODO: We could remove filtering on the merged_video to allow for multiple tests on each modality (using all videos partiitons)."""
        
    dset = get_dataset(dataset_name='base', root_dir=root_dir, scenario='present')
    df = dset.metadata.sort_values(['modality'], ascending=True).copy()
    #df = df[(df['video_name'] == 'merged_video') | (df['n_videos'] == 1)]

    results_save_path = os.path.join(get_data_root(), 'dataframes', 'persistent_metadata', 'cross_correlation_22022025.csv')
    if os.path.isfile(results_save_path):
        results = pd.read_csv(results_save_path)
    else:
        results = pd.DataFrame(columns=['participant_id'])
        
    print(f'Saving synchronisation results to {results_save_path}')
        
    for i, (participant_id, group) in enumerate(df.groupby(['participant_id'])):
        
        participant_id=participant_id[0]

        
        if participant_id in results.participant_id.tolist():
            result_row = results[results['participant_id'] == participant_id].iloc[0]
        else:
            result_row = None
            
        print(f'Starting to syncronize {participant_id}')
        
        result = synchronize(participant_id, group, result_row, verbose=verbose)
        results = pd.concat([results, pd.DataFrame(result, index=[0])], ignore_index=True, verify_integrity=False)
        results.to_csv(results_save_path, index=False)
        
        print(f'{i+1}/{len(df.participant_id.unique())} - {participant_id} done.')


    # Apply the check_constraints function to each row and add summary stats
    results['constraints_satisfied'] = results.apply(check_constraints, axis=1)
    results['max_distorsion'] = results.apply(compute_max_distorsion, axis=1)

    results['has_Tobii_audio'] = results.has_Tobii_audio.apply(lambda x: True if (not pd.isnull(x) and x) else False )
    results['has_GoPro1_audio'] = results.has_GoPro1_audio.apply(lambda x: True if (not pd.isnull(x) and x) else False )
    results['has_GoPro2_audio'] = results.has_GoPro2_audio.apply(lambda x: True if (not pd.isnull(x) and x) else False )
    results['has_GoPro3_audio'] = results.has_GoPro3_audio.apply(lambda x: True if (not pd.isnull(x) and x) else False )
    results['num_audio'] = results.apply(lambda x: np.nansum([x.has_GoPro1_audio, x.has_GoPro2_audio, x.has_GoPro3_audio, x.has_Tobii_audio]), axis=1)#np.nansum([x.has_GoPro1_audio, x.has_GoPro2_audio, x.has_GoPro3_audio, x.has_Tobii_audio], axis=1))
    results.to_csv(results_save_path, index=False)

    print('Done. Syncronisation results saved in {}'.format(results_save_path))
    
    return results

    
def synchronize(participant_id, group_df, result_row=None, verbose=False):
    """Compute offset between modalities of a participant using the max cross-correlation between audio tracks."""
    
    # 1) Extract audio files
    for video_path in group_df.video_path.tolist():
        print(f'Extracting audio from  {video_path}')
        extract_audio(video_path, verbose=verbose)
        print('Extraction done.')
        
    # 2) Init result dictionnary and audio paths
    result = {'participant_id': participant_id}
    modality_pairs = list(combinations(['GoPro1', 'GoPro2', 'GoPro3', 'Tobii'], 2))
    for mod1, mod2 in modality_pairs:
        result[f'{mod1}_{mod2}'] = np.nan
    audio_paths = group_df.video_path.map(lambda x: x.replace('.mp4', '.wav')).tolist()
    audio_pairs = list(combinations(audio_paths, 2))

    # 3) Iterate over pairs of audio files
    for audio_file1, audio_file2 in audio_pairs:
        
        # Modality 1 and modality 2 are taken from the folder directory (not dataframe modality which is fixed)
        mod1, mod2 = os.path.basename(os.path.dirname(audio_file1)),os.path.basename(os.path.dirname(audio_file2))

        if result_row is not None and not np.isnan(result_row[f'{mod1}_{mod2}']):
            print('[WARNING] {}-{}_{} sync is already performed.'.format(participant_id, mod1, mod2))
            continue
        
        print('Processing {}-{}'.format(audio_file1, audio_file2))
    

        audio1, audio2, sr, is_valid = load_audio(audio_file1, audio_file2, max_duration=5) #20 minutes of max duration to prevent overloading

        if not is_valid:
            continue
        
        print(f'Size of audio signals: {audio1.shape} - {audio2.shape}')
        # Compute the cross-correlation
        #cross_correlation = signal.correlate(audio1, audio2)
        cross_correlation = signal.correlate(audio1, audio2, mode='full') / (np.linalg.norm(audio1) * np.linalg.norm(audio2))
        
        # Find the lag at which the maximum correlation occurs in seconds
        lags = signal.correlation_lags(audio1.size, audio2.size, mode="full")
        lag = lags[np.argmax(cross_correlation)]/sr
        
            
                
        # Register result and plot the cross-correlation figure
        result[f'{mod1}_{mod2}'] = lag
        result[f'{mod2}_{mod1}'] = -lag
        result[f'has_{mod1}_audio'] = True; result[f'has_{mod2}_audio'] = True
        output_file = os.path.join(get_data_root(), 'outputs', 'cross-correlation-outputs-22022025', 'cross_correlation_{}_{}_{}'.format(participant_id, mod1, mod2))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if lag > 0:
            text = 'Audio track {} is delayed wrt {} by {:.2f}s'.format(mod2, mod1, lag)
        else:
            text = 'Audio track {} is in advance wrt {} by {:.2f}s'.format(mod2, mod1, lag)
        
        
        plt.figure(figsize=(20, 5))
        plt.plot(cross_correlation)
        plt.title('Cross-correlation - {}\n{}'.format(participant_id, text), weight='bold')
        plt.xlabel('Samples'); plt.ylabel('Cross-correlation')
        plt.savefig(output_file, bbox_inches='tight')
        print('Cross-correlation saved in {}'.format(output_file))

        if verbose:
            print(text)
            plt.show()
        else:
            plt.close()

    # Save results
    return result

def check_constraints(row, margin=1):
    
    # List of constraints to check 
    constraints = [
        np.isnan(row['GoPro1_GoPro2'] + row['GoPro2_GoPro3'] - row['GoPro1_GoPro3']) or abs(row['GoPro1_GoPro2'] + row['GoPro2_GoPro3'] - row['GoPro1_GoPro3']) <= margin, # Tobii out
        np.isnan(row['GoPro1_GoPro2'] + row['GoPro2_Tobii'] - row['GoPro1_Tobii']) or abs(row['GoPro1_GoPro2'] + row['GoPro2_Tobii'] - row['GoPro1_Tobii']) <= margin, # GP3 out
        np.isnan(row['GoPro3_Tobii'] + row['GoPro2_GoPro3'] - row['GoPro2_Tobii']) or abs(row['GoPro3_Tobii'] + row['GoPro2_GoPro3'] - row['GoPro2_Tobii']) <= margin, # GP1 out
        np.isnan(row['GoPro3_Tobii'] - row['GoPro1_Tobii'] + row['GoPro1_GoPro3']) or abs(row['GoPro3_Tobii'] - row['GoPro1_Tobii'] + row['GoPro1_GoPro3']) <= margin, # GP2 out
    ]

    return all(constraints)

def compute_max_distorsion(row):
    """Max constraints distorsion. Used to determine a reasonable syncronization threshold."""

    # List of constraints to check
    check_max_distorsion = [
        0 if np.isnan(row['GoPro1_GoPro2'] + row['GoPro2_GoPro3'] - row['GoPro1_GoPro3']) else  abs(row['GoPro1_GoPro2'] + row['GoPro2_GoPro3'] - row['GoPro1_GoPro3']), # Tobii out
        0 if np.isnan(row['GoPro1_GoPro2'] + row['GoPro2_Tobii'] - row['GoPro1_Tobii']) else abs(row['GoPro1_GoPro2'] + row['GoPro2_Tobii'] - row['GoPro1_Tobii']), # GP3 out
        0 if np.isnan(row['GoPro3_Tobii'] + row['GoPro2_GoPro3'] - row['GoPro2_Tobii']) else abs(row['GoPro3_Tobii'] + row['GoPro2_GoPro3'] - row['GoPro2_Tobii']), # GP1 out
        0 if np.isnan(row['GoPro3_Tobii'] - row['GoPro1_Tobii'] + row['GoPro1_GoPro3']) else abs(row['GoPro3_Tobii'] - row['GoPro1_Tobii'] + row['GoPro1_GoPro3']), # GP2 out
    ]

    return np.max(check_max_distorsion)

def load_audio(audio_file1, audio_file2, max_duration=20):
    
    # Load the audio files
    try:
        audio1, sr1 = librosa.load(audio_file1)
    except:
        print('[WARNING] {} not found'.format(audio_file1))
        return None , None, None, False
    try:
        audio2, sr2 = librosa.load(audio_file2)
    except:
        print('[WARNING] {} not found'.format(audio_file2))
        return None , None, None, False
    

    if sr1 != sr2:
        audio2 = librosa.resample(audio2, sr2, sr1)
        sr2 = sr1
    # # Ensure the sampling rates are the same
    # if sr1 != sr2:
    #     raise ValueError("The two audio files must have the same sampling rate.")
    
    
    # Calculate the number of samples that correspond to 20 minutes
    num_samples_20min = max_duration * 60 * sr1

    # Check if the audio is longer than 20 minutes
    if len(audio1) > num_samples_20min:
        # If it is, truncate it to the first 20 minutes
        audio1 = audio1[:num_samples_20min]

    # Do the same for the second audio
    if len(audio2) > num_samples_20min:
        audio2 = audio2[:num_samples_20min]
        
    is_valid = True

    return audio1, audio2, sr2, is_valid


def parse_args():
    
    parser = argparse.ArgumentParser(description='Audio-based modalities synchronisation')
    parser.add_argument('-r', '--root_dir', default='local', help='local or harddrive')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
            
    if args.root_dir == 'local':
        
        root_dir = None
        
    elif args.root_dir == 'harddrive':
        
        root_dir = '/Volumes/harddrive/data' 
        
    elif args.root_dir == 'Smartflat':
        
        root_dir = '/Volumes/Smartflat/data' 
        
    main(root_dir = root_dir)
    
    sys.exit(0)    