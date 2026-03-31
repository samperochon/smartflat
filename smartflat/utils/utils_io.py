import datetime
import logging
import os
import pickle
import socket
import subprocess
from glob import glob
from typing import Any

# TODO: weird that the three are imported only for assign_computation, maybe change the location...
import numpy as np
import pandas as pd
from decord import VideoReader, bridge, cpu
from IPython.display import display
from joblib import dump, load

try:
    from petrel_client.client import Client

    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False
bridge.set_bridge("torch")


from smartflat.configs.loader import get_complete_configs, import_config
from smartflat.constants import (
    AVAILABLE_ROUND_NUMBERS,
    LOCAL_MACHINE_NAMES,
    available_modality,
    available_tasks,
    enabled_modalities,
    hard_parsed_path,
    mapping_participant_id_fix,
    ordered_cluster_types,
)
from smartflat.utils.utils_coding import *


def get_data_root(machine_name=None, local=False):

    if machine_name is None:
        machine_name = socket.gethostname()

    if machine_name in LOCAL_MACHINE_NAMES:  # Port 8888
        if not local:
            DATA_ROOT = get_gold_data_path()
        else:
            DATA_ROOT = '/Users/samperochon/github-repositories/smartflat/data/data-gold-final'
                
    elif machine_name == "PCNomad-Borelli2":

        DATA_ROOT = "/media/sam/Smartflat/data-gold-final"

    elif machine_name == "pomme":  # Port 1111
        DATA_ROOT = "/home/perochon/data-gold-final"

    elif machine_name == "disque-dur":
        DATA_ROOT = "/home/perochon/data"

    elif machine_name == "asure":  # Port 3221
        DATA_ROOT = "/home01/sam/data"

    elif machine_name == "cheetah":  # Port 4372
        DATA_ROOT = '/diskA/sam_data/data-features'#"/diskA/sam_data/data-gold-final"  # '/home/sam/data'#'/dev/shm/samperochon/data'
        
    elif machine_name == "mate":  # Port 6740
        DATA_ROOT = "/home/sam/data"

    elif machine_name == "DESKTOP-PMRIG7H":
        DATA_ROOT = "C:\\Users\\admin\\samperochon\\data"

    elif "ruche" in machine_name or "node" in machine_name:

        DATA_ROOT = "/gpfs/workdir/perochons/data-gold-final"

    elif machine_name == "smartflat":  # Port 6740
        DATA_ROOT = "/Volumes/Smartflat/data"

    elif machine_name == "gold":

        DATA_ROOT = get_gold_data_path()
        
    elif machine_name == "light":

        DATA_ROOT = get_light_data_path()        

    else:
        print("Machine name: {} is unknown.".format(machine_name))
        raise ValueError
    return DATA_ROOT


def get_gold_data_path(machine_name=None):

    if machine_name is None:
        machine_name = socket.gethostname()

    if "cheetah" in machine_name:
        DATA_ROOT = "/diskA/sam_data/data-gold-final"  # '/home/sam/data'#'/dev/shm/samperochon/data'

    elif "ruche" in machine_name or "node" in machine_name:
        DATA_ROOT = "/gpfs/workdir/perochons/data-gold-final"

    elif "pomme" in machine_name:
        DATA_ROOT = "/home/perochon/data-gold-final"


    elif machine_name == "PCNomad-Borelli2":
        DATA_ROOT = "/media/sam/Smartflat/data-gold-final"

    elif machine_name in LOCAL_MACHINE_NAMES:  # Port 8888

        if os.path.isdir('/Volumes/Smartflat'):
            DATA_ROOT = "/Volumes/Smartflat/data-gold-final"  # HAS CHANGED #PATHGOLDDATASET
        else:
            DATA_ROOT = '/Users/samperochon/github-repositories/smartflat/data/data-gold-final'
    else:
        print("Machine name: {} is unknown.".format(machine_name))
        raise ValueError

    return DATA_ROOT

def get_light_data_path(machine_name=None):

    if machine_name is None:
        machine_name = socket.gethostname()

    if "cheetah" in machine_name:
        DATA_ROOT = "/diskA/sam_data/data-gold-light"  # '/home/sam/data'#'/dev/shm/samperochon/data'

    elif "ruche" in machine_name or "node" in machine_name:
        DATA_ROOT = "/gpfs/workdir/perochons/data-gold-light"

    elif "pomme" in machine_name:
        DATA_ROOT = "/home/perochon/data-gold-light"

    elif machine_name in LOCAL_MACHINE_NAMES:  # Port 8888

        DATA_ROOT = "/Volumes/Smartflat/data-gold-light"  # HAS CHANGED
    
    elif machine_name == "PCNomad-Borelli2":  # Port 1111
        DATA_ROOT = "/media/sam/Smartflat/data-gold-light"

    else:
        print("Machine name: {} is unknown.".format(machine_name))
        raise ValueError

    return DATA_ROOT

def get_api_root(machine_name=None):

    if machine_name is None:
        machine_name = socket.gethostname()

    if machine_name in LOCAL_MACHINE_NAMES:  # Port 8888
        # API_ROOT = "/Users/samperochon/Borelli/algorithms/smartflat"
        API_ROOT = "/Users/samperochon/github-repositories/smartflat"

    elif machine_name == "pomme":  # Port 1111
        API_ROOT = "/home/perochon/smartflat"

    elif machine_name == "cheetah":  # Port 4372
        API_ROOT = "/home/sam/smartflat"

    elif "ruche" in machine_name or "node" in machine_name:
        API_ROOT = "/gpfs/users/perochons/smartflat"

    elif machine_name == "PCNomad-Borelli2":  # Port 1111
        API_ROOT = "/home/sam/smartflat"

    else:
        print("Machine name: {} is unknown.".format(machine_name))
        raise ValueError

    return API_ROOT

def get_host_name():

    name = socket.gethostname()
    if "ruche" in name:
        return "ruche"
    elif "sjp" in name:
        return "smartflat"
    else:
        return name


# TODO
# logging.basicConfig(filename=os.path.join(get_data_root(), 'log','utils_io.log'),
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.DEBUG)

# logging.info("Memory footprint input/output operations log")
# logger = logging.getLogger('main')

def fetch_qualification_mapping(verbose=False):
    """Fetch the mapping between cluster types and the centroids indices, for each clustering config files 
    howting a centroid array in the experiment folder of the clustering config.
    
    Note: the config names for which we build the qualification mapping from crossing meta-cluster categories (for inference, see more details below)
    must appear in the list after the configs from which the centroids were used. 
    
    """
    
    # 1) Build the mapping for clustering experiment for which manual annotation are performed 
    clustering_config_names = ['SymbolicSourceCosineInferenceGoldConfig',
                               'SymbolicSourceEuclideanInferenceGoldConfig',
                               'SymbolicSourceFaissCInferenceGoldConfig',
                               'SymbolicSourceFaissEInferenceGoldConfig',
                               
                               'SymbolicSourceInferenceRefinementGoldConfig',

                               
                               ]#'ClusteringDeploymentKmeansConfig', 'ClusteringDeploymentKmeansRefinementTaskConfig', 'ClusteringDeploymentKmeansRefinementTaskI2Config', 'ClusteringDeploymentKmeansRefinementTaskI3Config', 'ClusteringDeploymentKmeansRefinementTaskI4Config']
    
    round_numbers = list(range(1, np.max([np.max(AVAILABLE_ROUND_NUMBERS[annotator_id]) for annotator_id in AVAILABLE_ROUND_NUMBERS.keys()]) + 1))
    qualification_mapping = {}
    for clustering_config_name in clustering_config_names:
        qualification_mapping[clustering_config_name] = {}
        
        for annotator_id in ['samperochon', 'theoperochon']:
            
            qualification_mapping[clustering_config_name][annotator_id] = {}
            
            for round_number in round_numbers:
                
                config = import_config(clustering_config_name)

                input_path = os.path.join(get_data_root(), 'dataframes', 'annotations', 'pigeon-annotations', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}_prototypes_K_100.csv')
                if os.path.isfile(input_path):
                    results = pd.read_csv(input_path, sep=';')
                else:
                    #red(f'Error: {input_path} not found.')
                    continue
                #purple(f'{input_path}: N= {len(results)}')
                qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}'] = results.groupby(['cluster_type']).n_cluster.agg(list).to_dict()
                if verbose:
                    pass#print('------------------------------------------------------------------------')
                    #blue('Qualification mapping for {} (annotator: {}, round: {}):'.format(clustering_config_name, annotator_id, round_number)) 
    # 2) Build the mapping for clustering experiment for which the centroids are concatenated from the previous clustering
    # clustering configs based on prototypes_array stored sequentially as part of a cross meta-cluster types 
    # of clustering (e.g. for inference on the whole data using cross-iteration prototypes)
    
    reconstructed_config_names = ['ClusteringDeploymentKmeansCombinationCompleteConfig', 
                                  'ClusteringDeploymentKmeansInferenceConfig']

    for clustering_config_name in reconstructed_config_names:
        
        config = import_config(clustering_config_name)
        
        # # 1) Get counts of cluster type per clustering 
        # # TODO: there should be a way to merge the two following steps
        # n_cluster_types = {config: {} for config in input_clustering_config_names}
        # for config_name, round_number in zip(input_clustering_config_names, round_numbers_list):
            
        #     for annotator_id in ['samperochon']:
                
        #         n_cluster_types[config_name][annotator_id] = {}
                
        #         input_config = import_config(config_name)
        #         input_path = os.path.join(
        #             get_data_root(), 'dataframes', 'annotations', 'pigeon-annotations',
        #             input_config.experiment_name, input_config.experiment_id,  annotator_id, f'round_{round_number}_prototypes_K_100.csv'
        #         )
        #         #green(f'prototypes annotation path: {input_path}')
        #         if os.path.isfile(input_path):
        #             df = pd.read_csv(input_path, sep=';')
        #         else:
        #             print(f'/!\ Error: {input_path} not found.')
        #             continue
                
        #         if verbose:
        #             print('------------------------------------------------------------------------')
        #             blue('Qualification mapping for {} (annotator: {}, round: {}):'.format(clustering_config_name, annotator_id, round_number)) 
                    
        #         # Count occurrences of each cluster type
        #         cluster_counts = df['cluster_type'].value_counts().to_dict()
        #         n_cluster_types[config_name][annotator_id][f'round_{round_number}'] = sort_cluster_types(cluster_counts)
        # df_counts = pd.DataFrame(n_cluster_types).fillna(0)
        
        # print('Display count: ')
        # display(df_counts)
        
        # 2) Create the qualification mapping to match the way centroids are concatenated together when building the model
        qualification_mapping[clustering_config_name] = {}
        
        
        for annotator_id in ['samperochon', 'theoperochon']:
            
            round_numbers_list = AVAILABLE_ROUND_NUMBERS[annotator_id]
            
            qualification_mapping[clustering_config_name][annotator_id] = {}

            for outer_round_number in round_numbers_list:
                
                qualification_mapping[clustering_config_name][annotator_id][f'round_{outer_round_number}'] = {}
                n = 0
                
                if clustering_config_name == 'ClusteringDeploymentKmeansCombinationCompleteConfig':
                    input_clustering_config_names, round_numbers_list, qualification_cluster_types_list = get_complete_configs(round_number=outer_round_number)

                elif clustering_config_name == 'ClusteringDeploymentKmeansInferenceConfig':
                    
                    input_clustering_config_names, round_numbers_list, qualification_cluster_types_list = get_complete_configs(round_number=outer_round_number, inference_mode=True)
                    # print(f"input_clustering_config_names:\n\t{input_clustering_config_names}\n"
                    #         f"round_numbers:\n\t{round_numbers}\n"
                    #         f"qualification_cluster_types:\n\t{qualification_cluster_types}")
                            
                for input_clustering_config_name, qualification_cluster_types, inner_round_number in zip(input_clustering_config_names, qualification_cluster_types_list, round_numbers_list):

                    if len(qualification_cluster_types) == 0:
                        raise ValueError(f'No qualification mapping are supposed to exist for {input_clustering_config_name} ?')
                    
                
                    if verbose:
                        #print('------------------------------------------------------------------------')
                        #print(f'--> {input_clustering_config_name}')
                        pass#blue('Creation: {} (config={} annotator: {}, round: {}):'.format(clustering_config_name, input_clustering_config_name, annotator_id, inner_round_number)) 
                    
                    
                    if input_clustering_config_name in qualification_mapping.keys() and f'round_{inner_round_number}' in qualification_mapping[input_clustering_config_name][annotator_id].keys():
                        
                    
                        for cluster_type in qualification_cluster_types:
                            
                            if cluster_type not in qualification_mapping[input_clustering_config_name][annotator_id][f'round_{inner_round_number}'].keys():
                                if verbose:
                                    pass#print(f'Cluster type {cluster_type} not found in {input_clustering_config_name}')
                                continue
                                
                            # Count number of centroids in this category for the query config 
                            n_centroids = len(qualification_mapping[input_clustering_config_name][annotator_id][f'round_{inner_round_number}'][cluster_type]) 
                        
                            if cluster_type not in qualification_mapping[clustering_config_name][annotator_id][f'round_{outer_round_number}'].keys():
                                qualification_mapping[clustering_config_name][annotator_id][f'round_{outer_round_number}'][cluster_type] = list(np.arange(n, n+n_centroids))
                            else:
                                qualification_mapping[clustering_config_name][annotator_id][f'round_{outer_round_number}'][cluster_type].extend(list(np.arange(n, n+n_centroids)))

                            if verbose:
                                pass#print(f'\t{cluster_type} K= {n_centroids}')
                            n += n_centroids
                            #print(qualification_mapping[clustering_config_name])
                            

                    else:
                        #raise ValueError(f'Qualification mapping not found for {input_clustering_config_name}')
                        # We need this option when sampling prototypes from the combination config holding all the centroids (no need to filter in this case
                        # the centroids added with the refinement procedure on lighter input space. 
                        #raise NotImplementedError('Not implemented yet')                
                        pass
                        #qualification_mapping[clustering_config_name][cluster_type] = df_counts[cluster_type].sum()
                        
            if verbose:
                pass#print('------------------------------------------------------------------------')
        
        # green('Qualification mapping for ClusteringDeploymentKmeansCrossingCombinationConfig:')
        # for ct, v in qualification_mapping['ClusteringDeploymentKmeansCrossingCombinationConfig'].items():
        #     print(f'\tCluster Type {ct} => {len(v)}')
        
        
        

    multiannot_config_names = ['ClusteringDeploymentKmeansMergeInferenceConfig']

    for clustering_config_name in multiannot_config_names:
        
        config = import_config(clustering_config_name)        
        
        # 2) Create the qualification mapping to match the way centroids are concatenated together when building the model
        qualification_mapping[clustering_config_name] = {}
        
        
        for annotator_id in ['fusionperochon']:
            
            round_numbers_list = AVAILABLE_ROUND_NUMBERS[annotator_id]
            
            qualification_mapping[clustering_config_name][annotator_id] = {}

            for outer_round_number in round_numbers_list:
                
                qualification_mapping[clustering_config_name][annotator_id][f'round_{outer_round_number}'] = {}
                n = 0
                
                input_clustering_config_names = config.model_params['input_clustering_config_names']
                qualification_cluster_types_list = config.model_params['qualification_cluster_types']
                round_numbers_list = config.model_params['round_numbers']
                annotator_ids = config.model_params['input_annotator_ids']
                
                
                for _annotator_id, input_clustering_config_name, qualification_cluster_types, inner_round_number in zip(annotator_ids, input_clustering_config_names, qualification_cluster_types_list, round_numbers_list):

                    if len(qualification_cluster_types) == 0:
                        raise ValueError(f'No qualification mapping are supposed to exist for {input_clustering_config_name} ?')
                    
                
                    if verbose:
                        #print('------------------------------------------------------------------------')
                        #print(f'--> {input_clustering_config_name}')
                        pass#blue('Creation: {} (config={} annotator: {}, round: {}):'.format(clustering_config_name, input_clustering_config_name, annotator_id, inner_round_number)) 
                    
                    
                    if input_clustering_config_name in qualification_mapping.keys() and f'round_{inner_round_number}' in qualification_mapping[input_clustering_config_name][_annotator_id].keys():
                        
                    
                        for cluster_type in qualification_cluster_types:
                            
                            if cluster_type not in qualification_mapping[input_clustering_config_name][_annotator_id][f'round_{inner_round_number}'].keys():
                                if verbose:
                                    pass#print(f'Cluster type {cluster_type} not found in {input_clustering_config_name}')
                                continue
                                
                            # Count number of centroids in this category for the query config 
                            n_centroids = len(qualification_mapping[input_clustering_config_name][_annotator_id][f'round_{inner_round_number}'][cluster_type]) 
                        
                            if cluster_type not in qualification_mapping[clustering_config_name][annotator_id][f'round_{outer_round_number}'].keys():
                                qualification_mapping[clustering_config_name][annotator_id][f'round_{outer_round_number}'][cluster_type] = list(np.arange(n, n+n_centroids))
                            else:
                                qualification_mapping[clustering_config_name][annotator_id][f'round_{outer_round_number}'][cluster_type].extend(list(np.arange(n, n+n_centroids)))

                            if verbose:
                                pass#print(f'\t{cluster_type} K= {n_centroids}')
                            n += n_centroids
                            #print(qualification_mapping[clustering_config_name])
                            

                    else:
                        #raise ValueError(f'Qualification mapping not found for {input_clustering_config_name}')
                        # We need this option when sampling prototypes from the combination config holding all the centroids (no need to filter in this case
                        # the centroids added with the refinement procedure on lighter input space. 
                        #raise NotImplementedError('Not implemented yet')                
                        pass
                        #qualification_mapping[clustering_config_name][cluster_type] = df_counts[cluster_type].sum()
                        
            if verbose:
                pass#print('------------------------------------------------------------------------')
        
        # green('Qualification mapping for ClusteringDeploymentKmeansCrossingCombinationConfig:')
        # for ct, v in qualification_mapping['ClusteringDeploymentKmeansCrossingCombinationConfig'].items():
        #     print(f'\tCluster Type {ct} => {len(v)}')
        
    
    # Add the symbolic represen
    qualification_mapping['SymbolicSourceInferenceCompleteGoldConfig'] = qualification_mapping['ClusteringDeploymentKmeansCombinationCompleteConfig']
    qualification_mapping['SymbolicSourceInferenceGoldConfig'] = qualification_mapping['ClusteringDeploymentKmeansInferenceConfig']
    qualification_mapping['SymbolicSourceMergeInferenceGoldConfig'] = qualification_mapping['ClusteringDeploymentKmeansMergeInferenceConfig']
    
    qualification_mapping['SymbolicSourcePrototypesTSInferenceGoldConfig'] = qualification_mapping['SymbolicSourceInferenceGoldConfig']



    if verbose:
        for clustering_config_name, q_annotator in qualification_mapping.items():
            
            for annotator_id, q_rounds in q_annotator.items():
                
                for round_number, q_cluster_types in q_rounds.items():
                                                    
                    n_tot_centroids = np.sum([len(v) for v in q_cluster_types.values()])
                    green('Qualification mapping for {} ({} round={} K={}):'.format(clustering_config_name,annotator_id, round_number, n_tot_centroids))
                    for ct, v in q_cluster_types.items():
                        print(f'\tCluster Type {ct} => {len(v)} - 5 first: {v[:5]}')
        
        if len(qualification_mapping.keys()) > 0:
            from smartflat.utils.utils_visualization import (
                plot_multiple_qualification_mapping,
            )

            plot_multiple_qualification_mapping(qualification_mapping, figsize=(25, 25))

        print('------------------------------------------------------------------------')
    # qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['foreground'] = qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['task-definitive'] + qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['exo-definitive']
    # qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['foreground'] = qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['task-definitive'] + qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['exo-definitive']
    # TODO/ add foreground at runtime ?
    # 
    # Re-order the clusters index
    # for config_name in qualification_mapping.keys():
    #     for cluster_type in qualification_mapping[config_name].keys():
            
    #         qualification_mapping[config_name][cluster_type] = sorted(qualification_mapping[config_name][cluster_type])

            
    return qualification_mapping

def sort_cluster_types(cluster_counts):
    return {k: cluster_counts.get(k, 0) for k in ordered_cluster_types}


def get_last_filename(filenames):
    """TODO: historicized some metadata files if needed.

    allpath = [ k for k in os.listdir(path_metadata_output) if k.startswith("metadata_")]
    path = os.path.join(get_Data_root(),get_last_filename(allpath))

    """
    last_filename = None
    last_date = None

    for filename in filenames:
        parts = filename.split("_")
        if len(parts) > 1:
            date_str = parts[-1].split(".")[0]  # Remove extension
            try:
                date = datetime.strptime(date_str, "%d-%m-%Y")
                if last_date is None or date > last_date:
                    last_date = date
                    last_filename = filename
            except ValueError:
                # Ignore files with invalid date format
                pass

    return last_filename


def save_df(o, path):
    dump(o, path) 

def load_df(path):
    return load(path)

# Video utils


def get_video_loader():

    def _loader(video_path):
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return vr

    return _loader


def check_video(video_path):
    """Check the video is valid."""
    try:
        vr = VideoReader(video_path, num_threads=0, ctx=cpu(0))
        return True
    except:
        return False


def extract_audio_files(process=False):
    """Extract audio (.wav) files of consolidated videos."""

    dset = get_dataset(dataset_name="base", scenario="present")
    df = dset.metadata.sort_values(["modality"], ascending=True).copy()
    df = df[(df["video_name"] == "merged_video") | (df["n_videos"] == 1)]

    for video_path in df.video_path.tolist():

        output_audio_path = os.path.join(
            os.path.dirname(video_path),
            os.path.basename(video_path).split(".")[0] + ".wav",
        )

        if not os.path.isfile(output_audio_path) and process:
            subprocess.run(["ffmpeg", "-i", video_path, output_audio_path])
            print("Extracted audio file: {}.".format(output_audio_path))


def extract_audio(video_path, verbose=False):
    """Extract audio (.wav) files from a videos."""

    output_audio_path = os.path.join(
        os.path.dirname(video_path), os.path.basename(video_path).split(".")[0] + ".wav"
    )

    if not os.path.isfile(output_audio_path):
        subprocess.run(["ffmpeg", "-i", video_path, output_audio_path])
        if verbose:
            print("Done. Extracted audio file: {}.".format(output_audio_path))


def load_pca(task_name, modality, n_components, whiten=True):

    if whiten:

        filename = f"pca_{task_name}_{modality}_{n_components}_w.pkl"
    else:
        filename = f"pca_{task_name}_{modality}_{n_components}.pkl"

    print(f"Load pca from: {filename}")
    filepath = os.path.join(get_api_root(), "api", "models", "artifacts", filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} not found.")

    # Load the PCA model from the file
    with open(filepath, "rb") as f:
        pca = pickle.load(f)
    return pca


# Write/Fetch utils
def collect_embeddings(
    df,
    output_root=None,
    output_type="video_representation",
    separate_outputs=True,
    process=False,
    command_output="cp",
):
    """Copy folder structure and computed representations in the data/dump folder.
    
    `separate_outputs`: if True, the output is copied to the data/dump folder, otherwise the output_root is used.
    
    Note: For now, we exlude all features extracted from low-resolution data (light folder). 
    separate_outputs`: if True, the output is copied to the data/dump folder, otherwise the output_root is used."""

    if output_root is None:
        output_root = get_data_root()

    if separate_outputs:

        dump_location = os.path.join(output_root, "dump", output_type)
        os.makedirs(dump_location, exist_ok=True)
    else:

        dump_location = output_root
    # Only transfer the validated ones.
    df_success = df[
        (df[f"{output_type}_computed"]) & (df[f"flag_{output_type}"] == "success") &
        (df[f"{output_type}_is_light"] == False)
    ]

    n_copied = 0
    commands = []
    for task, participant_id, modality, file_path in df_success[
        ["task_name", "participant_id", "folder_modality", f"{output_type}_path"]
    ].to_numpy():

        try:
            embed_folder = os.path.join(dump_location, task, participant_id, modality)
            os.makedirs(embed_folder, exist_ok=True)

        except OSError:
            print(f"Error: {embed_folder}")
            continue

        if os.path.isfile(os.path.join(embed_folder, os.path.basename(file_path))):
            # print(f'{os.path.join(embed_folder, os.path.basename(file_path))} already exists.')
            continue

        command = [command_output, file_path, embed_folder]

        if process:

            blue(" ".join(command))
            subprocess.run(command)

        else:
            yellow(" ".join(command))
            commands.append(command)

        # Copy flags if necessary (cp if enforced as opposed to the actual output, to convey the state of computation)
        if output_type in ["speech_recognition", "speech_representation"]:
            flag_path = fetch_flag_path(file_path, f"flag_{output_type}")

            command = [
                "cp",
                flag_path,
                os.path.join(dump_location, task, participant_id),
            ]

            if process:
                blue(" ".join(command))
                subprocess.run(command)
            else:
                yellow(" ".join(command))
                commands.append(command)

        else:

            flag_path = fetch_flag_path(file_path, f"flag_{output_type}")
            command = [
                "cp",
                flag_path,
                os.path.join(dump_location, task, participant_id, modality),
            ]
            if process:
                blue(" ".join(command))
                subprocess.run(command)
            else:
                yellow(" ".join(command))
                commands.append(command)

        # TODOREMOVE print("Copied {} to {}/ ...".format(representation, embed_folder))

        # logger.info(f'{__file__} - {__name__} - Copied {output_type} to {embed_folder}')
        n_copied += 1

    # Copy the compute log file# Deprecated. Done using rsync command in periodic updates (cleaner)
    # metrics_path = os.path.join(
    #     get_data_root(),
    #     "dataframes", 
    #     "{}_compute_time_{}_detection.csv".format(socket.gethostname(), output_type),
    # )
    # output_metric_path = os.path.join(
    #     dump_location,
    #     "{}_compute_time_{}_detection.csv".format(socket.gethostname(), output_type),
    # )
    # if process:
    #     subprocess.run(["cp", metrics_path, output_metric_path])
    # else:
    #     commands.append(["cp", metrics_path, output_metric_path])
    # logger.info(f'{__file__} - {__name__} - Done. Copied {n_copied} files.')
    # print("rsync -ahuvz {}:{} /Users/samperochon/Borelli/algorithms/data/".format(socket.gethostname(), dump_location))
    # print("rsync -ahuvz {} cheetah:/diskA/sam_data/data".format(dump_location))
    # print('rsync -ahuvz dump_cheetah dump_asure dump_mate pomme:/home/perochon/data/')
    # print("? mv 'dump_cheetah/*compute_time_video_representation_learning.csv' dump_asure/*compute_time_video_representation_learning.csv' dump_mate/*compute_time_video_representation_learning.csv' dump_pomme//*compute_time_video_representation_learning.csv' {}".format(os.path.join(get_data_root(), 'metrics')))

    return commands


def distribute_embeddings(
    input_metafolder,
    output_dir=None,
    process=False,
    prompt_user=True,
    ask_once=True,
    overwrite=True,
):
    """Copy the content of a source embedding folder to the machine data_root, creating subfolders if necessary.
    Works by iterating over the folder of the provided directory (e.g. cuisine, lego, sub-partitions, etc), then participant_id, then modality, then for each files present there (except videos), the embeddings are

    Notes: Set `prompt_user` to False to execute the command without prompting the user to take action.

    Usage: Script: copy A to B, conserving the existing structure (of the data organisation).

    `
    import smartflat.utils.utils_io as io

    input_metafolder = os.path.join(get_data_root(), 'dump')
    copy_file(input_metafolder)
    `
    TODO: Make the function more general to be used with arbitrary folders sub-directories list.
    """

    do_process = None
    n_copied = 0
    commands = []

    dump_folders = [
        path
        for path in glob(os.path.join(input_metafolder, "*"))
        if os.path.isdir(path)
    ]

    for dump_folder in dump_folders:
        tasks_paths = [
            path for path in glob(os.path.join(dump_folder, "*")) if os.path.isdir(path)
        ]

        for task_path in tasks_paths:

            participant_paths = [
                path
                for path in glob(os.path.join(task_path, "*"))
                if os.path.isdir(path)
            ]

            for participant_path in participant_paths:

                modality_paths = [
                    path
                    for path in glob(os.path.join(participant_path, "*"))
                    if os.path.isdir(path)
                ]

                for modality_path in modality_paths:

                    # apply function to the embedding folder
                    content_paths = [
                        path for path in glob(os.path.join(modality_path, "*"))
                    ]

                    for content_path in content_paths:

                        file_name, modality, participant_id, task = parse_path(
                            content_path
                        )

                        output_path = fetch_path(
                            task, participant_id, modality, output_dir=output_dir
                        )
                        os.makedirs(output_path, exist_ok=True)

                        # Ask if the user want to process at initialization Determine function behavior depending on user
                        if prompt_user:
                            print(
                                f"Copying content from input folder {content_path} to {output_path}?."
                            )
                            do_process = _prompt_user()

                            if ask_once:
                                prompt_user = False
                        else:
                            do_process = process

                        if do_process:
                            print(output_path, file_name)
                            if check_exist(output_path, file_name):
                                # print(f"Output file already exist. Overwrite={overwrite}")
                                # Command to be executed from a source embedding folder to a source embedding folder
                                if overwrite:
                                    subprocess.run(
                                        ["cp", "-r", content_path, output_path]
                                    )
                                    n_copied += 1
                            else:
                                subprocess.run(["cp", "-r", content_path, output_path])
                                n_copied += 1

                        else:
                            print(
                                f"Copying content from input folder {content_path} to {output_path}?."
                            )

        print(
            "Done. {} files copied in {} from {}".format(
                n_copied, get_data_root(), os.path.basename(dump_folder)
            )
        )
        n_copied = 0
    return


def send_videos_to_remote(df, hard_disk_root, remote_name, process=False):

    # if remote_name == 'pomme':
    #     remote_root = 'perochon@pomme:/home/perochon/data'

    # elif remote_name == 'cheetah':
    #     remote_root = 'cheetah:{}'.format()

    # elif remote_name == 'mate':
    #     remote_root = 'perochon@pomme:/home/perochon/data'

    # elif remote_name == 'sam_asure':
    #     remote_root = 'perochon@pomme:/home/perochon/data'

    remote_root = "{}:{}".format(remote_name, get_data_root(remote_name))

    n = 0
    for index, row in df.iterrows():
        local_path = os.path.join(
            hard_disk_root,
            row["task_name"],
            row["participant_id"],
            row["modality"],
            os.path.basename(row["video_path"]),
        )
        if not os.path.isfile(local_path):
            print("[WARNING] Missing file: {}".format(local_path))
        else:
            # Change path
            local_path = os.path.join(
                hard_disk_root.replace(" ", r"\ "),
                row["task_name"],
                row["participant_id"],
                row["modality"],
                os.path.basename(row["video_path"]),
            )
            remote_path = os.path.join(
                remote_root, row["task_name"], row["participant_id"], row["modality"]
            )
            command = f"rsync -auvzh {local_path} {remote_path}"

            if process:
                os.system(command)
                n += 1
            else:
                print(command)
    print("Done. Copied {} files to remote.".format(n))

    return


def get_free_space(folder):
    """Return folder free space in GB."""
    total, used, free = map(
        int, os.popen('df -k "' + folder + '"').readlines()[-1].split()[1:4]
    )
    return free / 1024 / 1024  # Convert to GB


def get_file_size_in_gb(file_path):
    size_in_bytes = os.path.getsize(file_path)
    size_in_gb = size_in_bytes / (1024 * 1024 * 1024)
    return size_in_gb


def print_nvidia_smi_output():
    try:
        # Run the nvidia-smi command
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Check if the command was successful
        if result.returncode == 0:
            print(result.stdout)  # Print the standard output
        else:
            print(
                f"Error: {result.stderr}"
            )  # Print the error output if the command failed

    except FileNotFoundError:
        print(
            "nvidia-smi command not found. Ensure that the NVIDIA drivers are installed and nvidia-smi is in your PATH."
        )


# def load_pickle(filename: str) -> Any:
#     """Function to load pickle file from disk."""
#     with open(filename, "rb") as f:
#         return pickle.load(f)


# def save_pickle(filename: str, data: Any):
#     """Function to save python object to a pickle file on disk."""
#     with open(filename, "wb") as f:
#         pickle.dump(data, f)


def fetch_output_path(video_path, model_name):
    """Provide output file name associated with a model for a given video sample."""
    video_name, _, _, _ = parse_path(video_path)
    # TODOFIX print(video_name, model_name)

    if model_name == "vit_giant_patch14_224":
        name = "video_representations_VideoMAEv2"
        return os.path.join(
            os.path.dirname(video_path), name + "_" + video_name + ".npy"
        )

    elif model_name == "whisperx":
        name = "speech_recognition_diarization_whisperx"
        return os.path.join(
            os.path.dirname(video_path), name + "_" + video_name + ".json"
        )

    elif model_name == "multilingual-e5-large":
        name = "speech_representations_multilingual"
        return os.path.join(
            os.path.dirname(video_path), name + "_" + video_name + ".npy"
        )

    elif model_name == "hand_landmarks_mediapipe":
        # FIXME: note that the GoPro rows are being populated while they don't have hand landmarks processing for now
        name = "hand_landmarks_mediapipe"
        return os.path.join(
            os.path.dirname(video_path), name + "_" + video_name + ".json"
        )

    elif model_name == "skeleton_landmarks_mediapipe":
        name = "skeleton_landmarks"
        return os.path.join(
            os.path.dirname(video_path), name + "_" + video_name + ".json"
        )

    elif model_name == 'hand_landmarks_mediapipe':
        #FIXME: note that the GoPro rows are being populated while they don't have hand landmarks processing for now
        name = 'hand_landmarks_mediapipe'
        return os.path.join(os.path.dirname(video_path), name + '_' + video_name + '.json')
    
    elif model_name == 'skeleton_landmarks_mediapipe':
        name = 'skeleton_landmarks'
        return os.path.join(os.path.dirname(video_path), name + '_' + video_name + '.json')
                
    elif model_name == 'tracking_hand_landmarks_v1':
        name = 'tracking_hand_landmarks'
        return os.path.join(os.path.dirname(video_path), name + '_' + video_name + '.json')
        
    else:
        raise NotImplementedError


def fetch_flag_path(output_path, flag_type):
    """TODO: refactor."""

    video_name, modality, participant_id, task = parse_path(output_path)

    # TOfix, integrate to parse_path
    data_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(output_path)))
    )

    if flag_type == "flag_speech_recognition":

        file_path = os.path.join(
            data_root,
            task,
            participant_id,
            f".{video_name}_speech_recognition_flag.txt",
        )

    elif flag_type == "flag_speech_representation":

        file_path = os.path.join(
            data_root,
            task,
            participant_id,
            f".{video_name}_speech_representation_flag.txt",
        )

    elif flag_type == "flag_video_representation":

        file_path = os.path.join(
            os.path.dirname(output_path), f".{video_name}_video_representation_flag.txt"
        )

    elif flag_type == "flag_hand_landmarks":

        file_path = os.path.join(
            os.path.dirname(output_path), f".{video_name}_hand_landmarks_flag.txt"
        )

    elif flag_type == "flag_tracking_hand_landmarks":

        file_path = os.path.join(
            os.path.dirname(output_path), f".{video_name}_tracking_hand_landmarks_flag.txt"
        )
        
    elif flag_type == "flag_skeleton_landmarks":

        file_path = os.path.join(
            os.path.dirname(output_path), f".{video_name}_skeleton_landmarks_flag.txt"
        )

    else:
        raise ValueError
    return file_path


def fetch_path(task, participant_id, modality, output_dir=None):
    """Fetch the embedding folder name of a modality for a participant"""
    if output_dir is not None:
        return os.path.join(output_dir, task, participant_id, modality)
    else:
        return os.path.join(get_data_root(), task, participant_id, modality)


def fetch_has_gaze(row, verbose=False):
    
    if row.modality != 'Tobii':
        return 0 
    
    # Define the path to search for gaze data
    gaze_data_list_1 = glob(
        os.path.join(
            os.path.dirname(get_data_root()),
            'data-gaze', '*', f'*{row.participant_id}*'
        )
    )
    
    reversed_mapping_participant_id = {v: k for k, v in mapping_participant_id_fix.items()}
    gaze_data_list_2 = glob(
        os.path.join(
            os.path.dirname(get_data_root()),
            'data-gaze', '*', '*{}*'.format(reversed_mapping_participant_id.get(row.participant_id, 'ABSENT_ID_FROM MAPPING'))
        )
    )
    
    gaze_data_list = set(gaze_data_list_1 + gaze_data_list_2)

    
    # Determine if gaze data is available
    if len(gaze_data_list) == 0:
        if verbose:
            print(f"No gaze data found for participant {row.participant_id}")
        return 0  # No gaze data
    elif len(gaze_data_list) == 1:
        if verbose:
            print(f"Found gaze data for participant {row.participant_id}: {list(gaze_data_list)}")
        return 1  # Gaze data available
    else:
        print(gaze_data_list)
        raise ValueError(f"Multiple gaze data files found for participant {row.participant_id}")


def parse_video_metadata(row):

    # if not row[['fps', 'n_frames', 'date', 'size']].isna().all() or (type(row.video_path) != str):
    #    return pd.Series(row[['fps', 'n_frames', 'date', 'size']].to_list())
    if not np.isnan(row.fps):
        return pd.Series([row.fps, row.n_frames, row.date, row.size])
    try:
        vr = VideoReader(row.video_path, num_threads=0, ctx=cpu(0))
    except:
        print(
            f"Error reading video: {row.video_path} - {row.task_name} - {row.participant_id} - {row.modality}"
        )

        return pd.Series([np.nan] * 4)
    fps = vr.get_avg_fps()
    n_frames = vr._num_frame
    size = get_file_size_in_gb(row.video_path)
    timestamp = os.path.getctime(row.video_path)
    date = datetime.datetime.fromtimestamp(timestamp)

    return pd.Series([fps, n_frames, date, size])


def parse_dir_name(path):

    modality, participant_id, task = (
        path.split("/")[-1],
        path.split("/")[-2],
        path.split("/")[-3],
    )
    return task, participant_id, modality


def parse_path(path):
    """Parse data_root/task/modality from an embedding folder path

    Example:
        video_name, modality, participant_id, task = parse_path(path)

    """
    # Check the black list of hard-coded paths
    # Unfortunately this is necessary as sthe way to map the anotations
    # To the actual administration is through looking at the original video path at the PErcy computer
    # Sometimes the path does not follow standardization and we have to hard-code the exceptions
    if path in hard_parsed_path:
        return hard_parsed_path[path]

    if os.path.basename(path).startswith("."):  #

        filename, modality, participant_id, task = (
            path.split("/")[-1].split(".")[1],
            path.split("/")[-2],
            path.split("/")[-3],
            path.split("/")[-4],
        )

        if "video_representation_flag" in filename:
            video_name = filename.split("_video_representation_flag")[0]

        elif "speech_recognition_flag" in filename:
            video_name = filename.split("_speech_recognition_flag")[0]

        elif "speech_representation_flag" in filename:
            video_name = filename.split("_speech_representation_flag")[0]

        elif "skeleton_landmarks_flag" in filename:
            video_name = filename.split("_skeleton_landmarks_flag")[0]
            
        elif "tracking_hand_landmarks_flag" in filename:
            video_name = filename.split("_tracking_hand_landmarks_flag")[0]
            

        elif "hand_landmarks_flag" in filename:
            video_name = filename.split("_hand_landmarks_flag")[0]

        else:
            video_name = filename

    else:
        try:
            filename, modality, participant_id, task = (
                path.split("/")[-1].split(".")[0],
                path.split("/")[-2],
                path.split("/")[-3],
                path.split("/")[-4],
            )
        except:
            print(f"Error parsing path: {path}")
            return np.nan, np.nan, np.nan, np.nan   

        if filename.startswith("video_representations"):
            video_name = (
                os.path.basename(path)
                .split("video_representations_VideoMAEv2_")[1]
                .split(".npy")[0]
            )

        elif filename.startswith("speech_recognition"):
            video_name = (
                os.path.basename(path)
                .split("speech_recognition_diarization_whisperx_")[1]
                .split(".json")[0]
            )

        elif filename.startswith("speech_representations"):
            video_name = (
                os.path.basename(path)
                .split("speech_representations_multilingual_")[1]
                .split(".npy")[0]
            )

        elif filename.startswith("skeleton_landmarks"):

            video_name = (
                os.path.basename(path).split("skeleton_landmarks_")[1].split(".json")[0]
            )
            
        elif filename.startswith("tracking_hand_landmarks"):
            
            try:
                video_name = (
                    os.path.basename(path)
                    .split("tracking_hand_landmarks_")[1]
                    .split(".json")[0]
                )
            except:
                red(path)
                red(filename)
                video_name='ERROR'
                


        elif filename.startswith("hand_landmarks"):
            video_name = (
                os.path.basename(path)
                .split("hand_landmarks_mediapipe_")[1]
                .split(".json")[0]
            )
                

        else:
            video_name = filename
            
        if task not in available_tasks:
            pass#rint(f'Error: {task} not in available tasks.')
        if modality not in available_modality:
            pass#print(f'Error: {modality} not in available modality.')

    return video_name, modality, participant_id, task


def parse_flag(output_path, modality, flag_type):
    """'success' or 'failure' or 'unprocessed' or 'disabled' if un-processed (missing flag)"""

    if flag_type == "flag_collate_video":
        filepath = os.path.join(
            os.path.dirname(output_path), ".merged_video_collate_flag.txt"
        )

        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                flag = f.readline().strip()
        else:
            flag = "unprocessed"
        return flag

    filepath = fetch_flag_path(output_path, flag_type)
    _, _, _, task_name = parse_path(output_path)

    #print('output_path, modality, flag_type', output_path, modality, flag_type)
    # Fixes
    task_name = (
        "cuisine"
        if ("cuisine" in task_name or "gateau" in task_name)
        else "lego" if "lego" in task_name else task_name
    )

    if modality not in enabled_modalities[task_name][flag_type]:
        return "disabled"

    if os.path.isfile(filepath):
        try:
            with open(filepath, "r") as f:
                flag = f.readline().strip()
        except:
            print(f"Error reading flag: {filepath}")
            # os.remove(filepath)
            flag = "corrupted"
        return flag
    else:
        return "unprocessed"  # np.nan


def parse_participant_id(string):
    """
    Parse participant ID strings into components.
    Supports various patterns including G104_P88_LABBen_01022023 and
    G159_RAYVia_SDS2_M12_V2_31052024_gateau.
    """
    components = string.split("_")
    
    # Handle specific known edge cases
    if string == 'SC_L171_PXX_240823':
        task_num, diag_num, trigram, date = 'L171', 'PXX', 'xxxxxx', '240823'

    # Handle formats with 6 components including "gateau" ending
    elif len(components) == 6:
        task_num, diag_num, trigram, period, visit, date = components[:6]
        trigram = trigram.lower().strip()
        date = date[:6]  # Extract date portion (exclude "gateau" or any ending)

    # Handle formats with 5 components (e.g., 'SC' prefix)
    elif len(components) == 5 and components[0] == "SC":
        task_num, diag_num, trigram, date = components[1:]
        trigram = trigram.lower().strip()

    # Handle formats with 4 components (standard case)
    elif len(components) == 4:
        task_num, diag_num, trigram, date = components
        trigram = trigram.lower().strip()
        
    elif len(components) == 7:
        task_num, diag_num, trigram, date = components[0], components[3], components[1], components[6]

    # Default case for unrecognized formats
    else:
        task_num, diag_num, trigram, date = np.nan, np.nan, np.nan, np.nan

    return pd.Series([task_num, diag_num, trigram, date])
    
def fetch_root_dir(output_path):
    return os.path.dirname(os.path.dirname(os.path.dirname(output_path)))

def parse_identifier(identifier):

    if identifier.startswith("SC_"):
        """

        Example usage:

        task_name, participant_id, modality, video_name = parse_identifier(identifier)
        """
        identifier = identifier[3:]

    # TOFIX
    # if identifier == 'BRASte_G124_240823_cuisine_Tobii_SC_BRASSte_G124_240823':
    #     return 'SC_BRASte_G124_240823', 'cuisine', 'Tobii', 'SC_BRASSte_G124_240823'
    # elif identifier == 'BRASte_G124_240823_cuisine_GoPro2_merged_video':
    #     return 'SC_BRASte_G124_240823', 'cuisine', 'GoPro2', 'merged_video'
    # Diviser l'identifiant en ses composants
    components = identifier.split("_")

    # Extraire les informations
    participant_id = "_".join(components[:4])
    task_name = components[4]
    modality = components[5]
    video_name = "_".join(components[6:])

    # Vérifier que la modalité est valide
    valid_modalities = ["GoPro1", "GoPro2", "GoPro3", "Tobii"]
    if modality not in valid_modalities:
        raise ValueError(
            f"Invalid modality: {modality}. Expected one of {valid_modalities}."
        )

    return task_name, participant_id, modality, video_name


def parse_task_number(row):
    try:
        return int(row[1:])
    except:
        return -1


def check_exist(output_path, file_name):
    return os.path.isfile(os.path.join(output_path, file_name))


def get_metadata(compute=False):
    """Create the metadata based on all remote metadata and of the hard disk in the local, with assignement and processing indexes."""

    if not compute:
        upath = os.path.join(
            get_data_root(), "dataframes/persistent_metadata/union_metadata.csv"
        )
        ipath = os.path.join(
            get_data_root(), "dataframes/persistent_metadata/inter_metadata.csv"
        )
        uniondf = pd.read_csv(upath)
        interdf = pd.read_csv(ipath)
        print("Loaded metadata.")
        return uniondf, interdf

    df = pd.concat(
        [
            pd.read_csv(path).assign(machine_name=os.path.basename(path).split("_")[0])
            for path in glob(
                os.path.join(get_data_root(), "dataframes/persistent_metadata/*")
            )
            if ("dataset_df" in path) and ("sjp49" not in path) and ("mate" not in path)
        ]
    )
    df = df.drop_duplicates(subset=["machine_name", "identifier"])

    # Propagate the video_representation_computed flag
    df["video_representation_computed"] = df.groupby("identifier")[
        "video_representation_computed"
    ].transform("any")

    df["in_pomme"] = df.groupby(["identifier"])["machine_name"].transform(
        lambda x: True if "pomme" in x.tolist() else False
    )

    # Quality and Control
    df["order"] = df.groupby(
        ["machine_name", "task_name", "participant_id", "modality"]
    )["date"].transform(
        lambda x: x.argsort() if pd.notnull(x).all() else [np.nan] * len(x)
    )
    df["n_videos"] = df.groupby(
        ["machine_name", "task_name", "participant_id", "modality"]
    )["identifier"].transform(lambda x: len(np.unique(x)))
    # df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    df["n_modality"] = df.groupby(["task_name", "participant_id"])[
        "modality"
    ].transform(lambda x: len(np.unique(x)))
    df = df.sort_values(["n_modality", "participant_id"], ascending=False)

    # Check if all videos within a modality have the same date
    df["unique_date"] = df.groupby(["task_name", "participant_id", "modality"])[
        "date"
    ].transform(lambda x: x.nunique() == 1)
    red(
        "Sanity check: number of rows with multiple dates: {} == 0 ?".format(
            len(df[~df["date"].isna() & ~df["unique_date"]])
        )
    )

    # Assign computation of and processing_gidx for each remote machines
    uniondf = assign_computation(df)

    # Revel all assigned videos that are not currently in the machine
    # TODO

    # Compute persistent_df
    uniondf["fps"] = uniondf.groupby("identifier")["fps"].transform(
        lambda x: x.dropna().mean()
    )
    uniondf["n_frames"] = uniondf.groupby("identifier")["n_frames"].transform(
        lambda x: x.dropna().mean()
    )
    # uniondf['machine_name'] = uniondf.groupby('identifier')['assignement'].transform('first')
    # uniondf['machine_name'] = uniondf.groupby('identifier')['assignement'].transform('first').fillna(uniondf['machine_name'])
    interdf = uniondf.drop_duplicates(subset=["identifier"])

    # Save dataframes
    upath = os.path.join(
        get_data_root(), "dataframes/persistent_metadata/union_metadata.csv"
    )
    ipath = os.path.join(
        get_data_root(), "dataframes/persistent_metadata/inter_metadata.csv"
    )
    uniondf.to_csv(upath, index=False)
    interdf.to_csv(ipath, index=False)

    yellow("mate")
    print(
        "scp {} mate:{} ".format(
            upath,
            os.path.join(get_data_root("mate"), "dataframes", "persistent_metadata"),
        )
    )
    print(
        "scp {} mate:{} ".format(
            ipath,
            os.path.join(get_data_root("mate"), "dataframes", "persistent_metadata"),
        )
    )
    yellow("cheetah")
    print(
        "scp {} cheetah:{} ".format(
            upath,
            os.path.join(get_data_root("cheetah"), "dataframes", "persistent_metadata"),
        )
    )
    print(
        "scp {} cheetah:{} ".format(
            ipath,
            os.path.join(get_data_root("cheetah"), "dataframes", "persistent_metadata"),
        )
    )
    yellow("asure")
    print(
        "scp {} asure:{} ".format(
            upath,
            os.path.join(get_data_root("asure"), "dataframes", "persistent_metadata"),
        )
    )
    print(
        "scp {} asure:{} ".format(
            ipath,
            os.path.join(get_data_root("asure"), "dataframes", "persistent_metadata"),
        )
    )
    yellow("pomme")
    print(
        "scp {} pomme:{} ".format(
            upath,
            os.path.join(get_data_root("pomme"), "dataframes", "persistent_metadata"),
        )
    )
    print(
        "scp {} pomme:{} ".format(
            ipath,
            os.path.join(get_data_root("pomme"), "dataframes", "persistent_metadata"),
        )
    )

    # TODO: automatize ?
    # command = 'scp /Users/samperochon/Borelli/algorithms/data/dataframes/persistent_metadata/union_metadata.csv pomme:/home/perochon/data/persistent_metadata'
    # os.system(command)
    return uniondf, interdf


def assign_computation(df):
    """Given the state of video storage + embeddding computation of each remote machine, define the assignement and order of processing of each administration for each assigned machine.

    Note: The assignement is done per video, determined only when the embedding computation is missing,
    and given to cheetah first, then mate, then asure, then pomme. If existing nowhere, by default to pomme to store the data (cheetah and mate memory are monitored).
    IDEA: maybe you can set assignement to NAN and manually control the data flow.

    FIXME: the assignement don't account for presence e.g. only in pomme, and the overhead associated... maybe it's okay as group number is long. Not too dynamic regime when working
    """

    # 1) Define assignement
    output_df = []
    for (identifier), group in df.groupby(["identifier"]):

        # REMOVE print("Processing task: {} for participant: {}".format(group.task.iloc[0], group.participant_id.iloc[0])) if verbose else None

        if group.video_representation_computed.any():
            output_df.append(
                group.assign(assignement=np.nan, distributed=group.in_pomme.any())
            )

        elif "cheetah" in group.machine_name.values:
            output_df.append(group.assign(assignement="cheetah", distributed=True))

        elif "mate" in group.machine_name.values:
            output_df.append(group.assign(assignement="mate", distributed=True))

        elif "asure" in group.machine_name.values:
            output_df.append(group.assign(assignement="asure", distributed=True))

        elif "pomme" in group.machine_name.values:
            output_df.append(group.assign(assignement="pomme", distributed=True))

        else:  # Video only in the disk
            output_df.append(group.assign(assignement="cheetah", distributed=False))

    outputdf = pd.concat(output_df)
    outputdf["processing_gidx"] = outputdf.groupby(
        ["assignement", "task_name", "participant_id"]
    ).ngroup()  # TODO: add date when enabled
    outputdf["processing_gidx"].replace(-1, np.nan, inplace=True)
    return outputdf


# Logged functions for api


def remove_videos(video_paths, process=False):
    """Remove video files."""
    n = 0
    size = 0
    shown = False
    for i, video_path in enumerate(video_paths):

        size += get_file_size_in_gb(video_path)
        command = "rm {}".format(video_path)

        if process:
            os.system(command)
            n += 1
        elif not shown:
            print("{} ... ({} files) ?".format(command, len(video_paths)))
            shown = True
        else:
            pass

    print("Total size of the files: {:.2f} Go.".format(size))
    print("Done. Removed {} video_files to remote.".format(n))


def change_names_in_modality(modality_path, mapping_names, process=False):

    commands = []
    files = glob(os.path.join(modality_path, "*")) + glob(
        os.path.join(modality_path, ".*")
    )
    for file in files:

        for old_name, new_name in mapping_names.items():

            if old_name in file:
                commands.append(["mv", file, file.replace(old_name, new_name)])

    if not process:
        for command in commands:
            print(" ".join(command))
            print("\n")

    else:
        for command in commands:
            logging.info(f"{__file__} - {__name__} - {command}")
            subprocess.run(command)
    return commands


def _prompt_user():
    """Copy file from the source to the destination folder, with prompting"""

    while True:
        user_input = input("Do you want to proceed? (y/n): ").lower()

        if user_input == "y":

            print("You chose to proceed.")
            do_process = True
            break

        elif user_input == "n":
            print("You chose not to proceed.")
            do_process = False
            break

        elif user_input == "q":
            # TODO: quit the program
            do_process = False
            break

        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    return do_process


def open_folder(odf, n_max=10):
    n = 0
    for i, row in odf.iterrows():
        n += 1
        if n > n_max:
            continue
        subprocess.run(["open", row["folder_path"]])


def get_video_output_paths(video_name, modality_folder):

    paths = []
    for root, dir, filepaths in os.walk(modality_folder):

        for filepath in filepaths:

            if video_name in filepath:

                paths.append(os.path.join(modality_folder, filepath))
    return paths
    return paths
