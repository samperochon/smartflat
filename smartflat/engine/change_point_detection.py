import argparse
import ast
import multiprocessing
import os
import sys
import time
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns
from scipy.spatial.distance import pdist

#from typing import Any, Callable, Dict, Literal, Optional



from smartflat.configs.loader import import_config, load_config
from smartflat.datasets.loader import get_dataset
from smartflat.engine.builders import build_metrics, build_model
from smartflat.utils.utils import get_expids, pairwise
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_io import get_data_root, load_df, parse_video_metadata, save_df

# 1) Experiment
def main(
    config_name="ChangePointDetectionConfig",
    experiment_id: str = None,
    model_params: dict = None,
    dataset_params: dict = None,
    whiten=None,
    results=None,
    continue_process=False,
    n_cpus="all",
    check_results=True,
    do_training=True, 
    n_subset = None,
):

    # Get experiment config
    config = import_config(config_name)
    

    if dataset_params is not None:
        config.dataset_params.update(dataset_params)

    if model_params is not None:
        config.model_params.update(model_params)

    if experiment_id is not None:
        config.experiment_id = experiment_id

    elif config.experiment_id is None:
        config.experiment_id = get_expids()

    if whiten is not None:
        config.whiten = whiten
        
    annotator_id = config.dataset_params["annotator_id"]
    round_number = config.dataset_params["round_number"]
    # Check already done
    if check_results:
        if results is None:
            results = get_results_change_point_detection(config, use_stored=False)

        # Remove already done TODOFIX : long term, you want to make this dpeendent of the experiment (e..g do we filter per n_pca_component?)
        results_exp = results[
            np.isclose(results["penalty"], config.model_params["penalty"])
            &
            # ((results['n_pca_components'] == config.dataset_params['n_pca_components']) | ( (config.dataset_params['n_pca_components'] is None) and (results['n_pca_components'].isna()))) &
            (results["task_name"] == config.dataset_params["task_names"][0])
            & (results["modality"] == config.dataset_params["modality"][0])
            # (results['whiten'] == config.dataset_params['whiten'])
        ]

        identifier_processed = results_exp.identifier.unique()
        experiment_id = results_exp.experiment_id.unique()
        print(f"Found {len(experiment_id)} experiment ids.")

        # New experiment
        if len(experiment_id) == 0:

            print('-> Running {}-{} with D={} and whiten={} and penalty={}.'.format(config.dataset_params['task_names'][0], 
                                                                                                                config.dataset_params['modality'][0], 
                                                                                                                config.dataset_params['n_pca_components'], 
                                                                                                                config.dataset_params['whiten'], 
                                                                                                                config.model_params['penalty']
                                                                                                                ))

        elif len(experiment_id) == 1:
            print('-> Experiment {}-{} with D={} and whiten={} and penalty={} aleady exists with {} ids.'.format(config.dataset_params['task_names'][0], 
                                                                                                                config.dataset_params['modality'][0], 
                                                                                                                config.dataset_params['n_pca_components'], 
                                                                                                                config.dataset_params['whiten'], 
                                                                                                                config.model_params['penalty'],
                                                                                                                results_exp.identifier.nunique()
                                                                                                                
                                                                                                                ))
            if continue_process:
                print(f"Continuing experiment with ID: {experiment_id[0]}")
            else:
                return 

            config.experiment_id = experiment_id[0]

        # else:
        #     print(f"Multiple experiments found: {experiment_id}. Exiting")
        #     return#raise ValueError(f"Multiple experiments found: {experiment_id}")

    else:
        identifier_processed = []
        
    #TODO: remove this after check 
    #output_folder = os.path.join(get_data_root(), 'experiments'); os.makedirs(output_folder, exist_ok=True)
    #output_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name); os.makedirs(output_folder, exist_ok=True)
    output_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)

    print(f'Saving outputs to {output_folder}.')
    filename = os.path.join(output_folder, "config.json")
    config.to_json(filename); #pprint(config.to_dict())

    # Get dataset
    dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)
    
    if do_training:
        dset.metadata = dset.metadata[dset.metadata['split'] == 'train']
    
    if n_subset is not None:
        dset.metadata = dset.metadata.sort_values('identifier').sample(n_subset, random_state=42)
        
    if config.model_params['kernel'] == 'rbf' and 'gamma' not in dset.metadata.columns:
         dset.load_gamma(config_name=config_name,  annotator_id=annotator_id, round_number=round_number, temperature=config.model_params.get('temperature', 1.0))
        
    # Remove missing cpts
    n = dset.metadata.identifier.nunique()
    dset.metadata.dropna(subset=['fps', 'n_frames'], inplace=True); print(f'/!\ {n - dset.metadata.identifier.nunique()}/{n} identifers discarded because of missing fps.')

    # Check if using personalized regularisation parameters
    
    if config.model_params['penalty'] == 'personalized':
        print(f'Loading personalized lambdas for {config.config_name}.')
        if 'lambda_1' in dset.metadata.columns:
            print(f'Personalized lambdas already loaded from experiment (trajectory CPD).')
        
        else:
            dset.load_optimal_lambdas(config.config_name, annotator_id=annotator_id, round_number=round_number)
        n = len(dset)
        display(dset.metadata.head(2))
        #dset.metadata.dropna(subset=[config.experiment_id], inplace=True)
        print(f'/!\ {n - len(dset)} identifers discarded because of missing personalized penalty.')

    # Check if using personalized regularisation parameters
    elif config.model_params['penalty'] == 'modality-specific':
        print(f'Loading personalized lambdas for {config.config_name}.')
        if 'lambda_1' in dset.metadata.columns:
            print(f'Personalized lambdas already loaded from experiment (trajectory CPD).')
        else:
            dset.load_optimal_lambdas(config.config_name, annotator_id=annotator_id, round_number=round_number)
        n = len(dset)
        #dset.metadata.dropna(subset=[config.experiment_id], inplace=True)
        display(dset.metadata.head(2))
        #dset.metadata.dropna(subset=[config.experiment_id], inplace=True)
        print(f'/!\ {n - len(dset)} identifers discarded because of missing calibrated penalty.')
        
    elif  config.model_params['penalty'] == 'slope_heuristics':
        print(f'Using the slope heuristics with slope={config.model_params["global_slope_heuristics"]}')
        
    else:
        print(f'Using global penalty={config.model_params["penalty"]}.')
        
    n = dset.metadata.identifier.nunique()
    dset.metadata = dset.metadata[~dset.metadata['identifier'].isin(identifier_processed)]
    print(
        f"/!\ {len(identifier_processed)} identifers discarded because already processed (check_results={check_results})."
    )

    dset.metadata.drop_duplicates(["identifier"], inplace=True)
    if len(dset) == 0:
        print(f"No data left to process. Exiting.")
        return
    # Perform multiprocessing change-point-detection
    
    print('Running change-point-detection for {} samples.'.format(len(dset)))
    multiprocess_change_point_detection(dset, config, n_cpus=n_cpus)


def main_deployment(
    config_name="ChangePointDetectionDeploymentConfig",
    annotator_id="samperochon",
    round_number=0,
    prototypes_config_name='SymbolicSourceInferenceGoldConfig',
    do_training=True,
    n_cpus="all",
):
    """Expect a deployment config."""

    config = import_config(config_name)
    experiment_ids = config.experiment_ids
    
    if prototypes_config_name is None:
        dataset_params={
                'annotator_id': annotator_id,
                'round_number': round_number,
                }
    else:
        dataset_params={
                'annotator_id': annotator_id,
                'round_number': round_number,
                'config_name': prototypes_config_name,

                }

    t_init = time.time(); t1 = time.time()
    for expriment_id in config.experiment_ids:


        main(
            config_name=config.config_name,
            experiment_id=expriment_id,
            dataset_params=dataset_params,
            n_cpus=n_cpus,
            do_training=do_training,
            check_results=False,  # We check dowstream on the existence of the change-point array.
        )

        print(f'Finished change-point detection for {expriment_id} in {time.time()-t_init:.2f} min.')
        t1 = time.time()

    print(f'Finished cpts for all expriment_ids in {(time.time() - t_init)/60:.2f} min.')

def change_point_detection(dset, from_idx, to_idx, config):
    """Apply change-point-detection to the dataset."""

    # Init. output folder
    output_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, config.dataset_params['annotator_id'], f'round_{config.dataset_params["round_number"]}')

    # Set minimum segment duration
    # try:
    min_size = config.model_params['min_segment_duration'] 
    # except: #-> Check the exception for that !
    #    print(f'[Warning] {idx} miss fps or n_frames')
    #    continue
        

    idx_list = list(range(from_idx, to_idx+1))

    if idx_list[-1] == len(dset):
        idx_list = idx_list[:-1]

    for i in idx_list:     
        row = dset.metadata.iloc[i]
        X, _, _ = dset[i] # Use potential transform
        identifier = row.identifier
        output_path = os.path.join(output_folder, f'{identifier}_cpts.npy')

        if os.path.isfile(output_path):
            print(f"{output_path} already exists.")
            #continue
            
            
        # Estimate gamma if necessary
        if config.model_params['kernel'] == 'rbf':
            
            # temperature = config.model_params.get('temperature', 1.0)
            # condensed_sq_dists = pdist(X, metric='sqeuclidean')
            # sigma_rbf = np.median(np.sqrt(condensed_sq_dists))
            # gamma= 1 / (2 * temperature * sigma_rbf ** 2)
            config.model_params['gamma'] = row.gamma
            #print(f'Computed gamma_rbf: {row.gamma:.2f}')
        
        # Build model
        model = build_model(config.model_name, config.model_params) 
        
        N, _ = X.shape
        if config.model_params['penalty'] == 'personalized':

            assert config.experiment_id in dset.metadata.columns, f'Column {config.experiment_id} not found in metadata.'

            penalty = row[config.experiment_id] / np.log(N)
            
            cpt = model.fit_predict(X, pen=penalty)[:-1]

        elif config.model_params['penalty'] == 'modality-specific':

            assert config.experiment_id in dset.metadata.columns, f'Column {config.experiment_id} not found in metadata.'

            penalty = row[config.experiment_id] / np.log(N)
            
            cpt = model.fit_predict(X, pen=penalty)[:-1]
            
        elif config.model_params['penalty'] == 'calibration':
            
            n_bkps = config.model_params['n_cpts']
            #try:
            cpt = model.fit_predict(X, n_bkps=n_bkps)[:-1]
            #except:
            #    print(X)
            #    print(f'Failed segmentation for {row.participant_id} X={X.shape} n_bkts={n_bkps}')
            #    cpt = []
                
        elif config.model_params['penalty'] == 'slope_heuristics':
            
            slope = config.model_params['global_slope_heuristics'] 
            penalty = - 2 * slope * N 
            
            cpt = model.fit_predict(X, pen=penalty)[:-1]
                
        else:
            
            penalty = config.model_params['penalty']/np.log(N)
            
            cpt = model.fit_predict(X, pen=penalty)[:-1]

        

        # Deprecated
        # if self.outliers_removed and remove_outliers:
        #    from_to_dict = retrieve_mapping(original_size=len(self.idx_embedding), mask_outliers=self.mask_outliers)
        #    self.cpt = [from_to_dict[c] for c in self.cpt]
        # Add ruptures of the outliers
        # outliers_cpt = list(np.argwhere(np.abs(np.ediff1d(self.mask_outliers))>0).flatten())
        # cpt  = sorted(list(set(self.cpt + outliers_cpt)))
        # Remove segments of size less than half a secont (artifact from the removal of outliers frames)
        # min_size=np.ceil(row.fps*.4/config.delta_t).astype(int)
        # cpt =  [j for i, j in pairwise(cpt) if j-i > min_size]
        # print("Total {} ruptures after pruning".format(len(cpt)-2))

        cpt = [0] + cpt + [N]
        
        n_bkps = len(cpt)-2

        #print("{}: {} ruptures, duration={:.2f}m min penalty={:.2f}".format(identifier, len(cpt)-2, row.n_frames / row.fps / 60 , penalty))
        print("{}: {} ruptures, duration={:.2f}m min n_cpts={:.2f}".format(identifier, len(cpt)-2, row.n_frames / row.fps / 60 , n_bkps))
        # Deprecated
        # self.mask_outliers[self.cpt[:-1]] = 1; self.idx_outliers = np.argwhere(self.mask_outliers==1).squeeze()

        np.save(output_path, np.array(cpt))

    return from_idx, to_idx

def multiprocess_change_point_detection(dset, config, n_cpus=8):
    
    if n_cpus == 'all':
        n_cpus = multiprocessing.cpu_count()
    elif n_cpus > multiprocessing.cpu_count():
        raise ValueError(f"Number of CPUs {n_cpus} is greater than the number of available CPUs {multiprocessing.cpu_count()}")
    
    pool = multiprocessing.Pool(n_cpus)
    N = len(dset.metadata)
    
    if n_cpus > N:
        n_cpus = N
        
    N_per_process = N / n_cpus
    
    print("Number of processes: " + str(n_cpus))
    print("Number of samples: " + str(N))
    
    # each task specifies a range of samples
    tasks = []
    for num_process in range(1, n_cpus + 1):
        if num_process == 1:
            start_index = (num_process - 1) * N_per_process 
        else:
            start_index = (num_process - 1) * N_per_process + 1
        end_index = num_process * N_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        tasks.append((dset, start_index, end_index, config)) #ARGUMENTS DE NOM_FONCTION
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process sample " + str(start_index))
        else:
            print(
                "Task #"
                + str(num_process)
                + ": Process administration-modality "
                + str(start_index)
                + " to "
                + str(end_index)
            )

    # start tasks
    results = []
    for i, t in enumerate(tasks):
        #try:
        results.append(pool.apply_async(change_point_detection, t))
        #except Exception as e:
        #    print(f"Une exception a été levée lors de l'exécution de change_point_detection {i} : {e}")

    for i, result in enumerate(results):
        #try:
        result.get()
        #except Exception as e:
        #    print(f"Une exception a été levée lors de l'obtention du résultat {i}: {e}")

    pool.close()

    return 


# 2) Results

def get_results_change_point_detection(config, annotator_id=None, round_number=None, experiment_folders=None, aggregate_per_experiment=False, use_stored=True):


    if annotator_id is None:
        annotator_id = config.dataset_params['annotator_id']
    if round_number is None:
        round_number = config.dataset_params['round_number']
    
    if experiment_folders is None:

        if config.experiment_id is None:
            experiment_folders = [os.path.join(get_data_root(), 'experiments', config.experiment_name)]
            results_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, annotator_id, f'round_{round_number}'); os.makedirs(results_folder, exist_ok=True)
        else:
            experiment_folders = [os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}')]
            results_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(results_folder, exist_ok=True)
    

    results_df_path = os.path.join(results_folder, "results.pkl")

    if not use_stored: # From scratch 

        dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)
        
        n = len(dset)
        display(dset.metadata[dset.metadata['duration'].isna()][['identifier', 'task_name', 'modality', 'participant_id', 'duration', 'video_path']])
                
        dset.metadata.dropna(subset=['duration'], inplace=True)
        print(f'/!\ {n - len(dset.metadata)} identifers discarded because of missing duration.')
        #n_list = [np.load(p).shape[0] for p in dset.metadata.video_representation_path.to_list()]
        n_list = dset.metadata.test_bounds.apply(lambda x: x[1] - x[0]).to_list()
         
        exps = []
        for experiment_folder in experiment_folders:
            exps.extend(
                [
                    dirpath
                    for dirpath in glob(os.path.join(experiment_folder, "*"))
                    if os.path.isdir(dirpath)
                ]
            )
        print(f'Found a total of {len(exps)} experiments folders in {experiment_folders}.')
        results = []
        for exp in exps:
            try:
                config = load_config(os.path.join(exp, annotator_id, f'round_{round_number}',  'config.json'))
            except:
                continue
            # FIX
            if 'normalization' not in config.dataset_params.keys():
                print('[WARNING deprecated] Adding normalization to config.dataset_params.')
                config.dataset_params['normalization'] = 'Z-score'

            if config.model_params["penalty"] in ["personalized", "modality-specific"]:

                if "lambda_1" not in dset.metadata.columns:
                    print(f"Load personalized lambdas for {config.config_name}.")
                    dset.load_optimal_lambdas(config.config_name, annotator_id=annotator_id, round_number=round_number)
                else:
                    print(f"Use loaded personalized lambdas for {config.config_name}.")
                penalty = dset.metadata.lambda_1.to_list()

            else:
                penalty = config.model_params["penalty"]

            results.append(
                pd.DataFrame(
                    {
                        "annotator_id": config.dataset_params["annotator_id"],
                        "round_number": config.dataset_params["round_number"],
                        "experiment_id": config.experiment_id,
                        "identifier": dset.metadata.identifier.to_list(),
                        "n_frames": dset.metadata.n_frames.to_list(),
                        "fps": dset.metadata.fps.to_list(),
                        "task_name": dset.metadata.task_name.to_list(),
                        "participant_id": dset.metadata.participant_id.to_list(),
                        "split": dset.metadata.split.to_list(),
                        "modality": dset.metadata.modality.to_list(),
                        "video_representation_path": dset.metadata.video_representation_path.to_list(),
                        "folder_modality": dset.metadata.folder_modality.to_list(),
                        "N": n_list,
                        "normalization": config.dataset_params["normalization"],
                        "kernel": config.model_params["kernel"],
                        "n_cpts": config.model_params["n_cpts"],
                        "n_pca_components": (
                            config.dataset_params["n_pca_components"]
                            if "n_pca_components" in config.dataset_params.keys()
                            else -1
                        ),
                        "whiten": (
                            config.dataset_params["whiten"]
                            if "whiten" in config.dataset_params.keys()
                            else False
                        ),  # TOFIX: not exactly stable in the long run unless all conigs have the whiten.
                        "model_name": config.model_name,
                        "penalty": penalty,
                        "cpts": dset.metadata.identifier.apply(
                            lambda x: (
                                np.load(os.path.join(exp,annotator_id, f'round_{round_number}',  f"{x}_cpts.npy"))
                                if os.path.isfile(os.path.join(exp, annotator_id, f'round_{round_number}', f"{x}_cpts.npy"))
                                else np.nan
                            )
                        ),
                    }
                )
            )
            


        if len(results) == 0:
            return pd.DataFrame()

        results = pd.concat(results)
        green(f'Collected {len(results)} results from {len(exps)} experiments.')
        if annotator_id is not None:
            results = results[results.annotator_id == annotator_id]
            print(f'Filtering results for annotator_id={annotator_id} -> {len(results)} results.')
        results['duration'] = results['n_frames'] / results['fps'] / 60
        results['n_cpts'] = results['cpts'].apply(lambda x: len(x) if type(x) == np.ndarray else np.nan).to_list()
        results['cpts_frequency'] = results['n_cpts'] / results['duration']
        if 'penalty' in results.columns and not (results.penalty.iloc[0]) == str:
            results['penalty'] = np.nan
            results['penalty_n'] = results['penalty'] / np.log(results['N']) 
            results['log_penalty'] = np.log(results['penalty'])
            

        results.n_pca_components.fillna(-1, inplace=True)
        results.whiten.fillna(-1, inplace=True)

        print('Initial length of results: ', len(results))

        # Remove outliers
        n = results.identifier.nunique()
        #print('Rows without any cpts')
        #print(results[results['cpts'].isna()].identifier.unique())
        results.dropna(subset=['cpts'], inplace=True); print(f'/!\ {n - results.identifier.nunique()}/{n} identifers discarded because of missing cpts (desired to retrieve results).'); n = results.identifier.nunique()
        #results = results[(results['fps'] > 10) & (results['duration'] > 20) & (results['duration'] < 90)];print(f'/!\ {n - results.identifier.nunique()}/{n} identifers discarded because of fps and duration criteria.')

        # Add bins label
        results['n_frames_binned'], bins_label = pd.cut(results['duration'], bins=4, retbins=True, labels=False)
        results['duration_labels'] = results.n_frames_binned.map({i: j for i, j in enumerate(['{}-{}'.format(int(s), int(e)) for s,e in pairwise(bins_label)]) })
        Ns=results.drop_duplicates(subset=['identifier']).groupby(['task_name', 'modality'])['duration_labels'].value_counts()
        results['duration_labels_counts'] = results.apply(lambda x: f'{x.duration_labels} (N={Ns.loc[x.task_name].loc[x.modality].loc[x.duration_labels]})', axis=1)

        # Convert arrays to string and save results
        #results['cpts'] = results['cpts'].apply(lambda x: np.array2string(x, separator=',') if type(x) == np.ndarray else np.nan)

        save_df(results, results_df_path)
        print(f'Saved results pickle file to {results_df_path}.')
        #results['cpts'] = results['cpts'].apply(lambda x: ast.literal_eval(x) if type(x) == str else np.nan)

    elif not os.path.isfile(results_df_path):
        raise ValueError(
            f"No results found at {results_df_path}. Please set use_stored=False to generate the results."
        )

    else:
        #results = pd.read_csv(results_df_path, low_memory=False)
        results = load_df(results_df_path)
        print(f'Loading results from {results_df_path}: {results.shape}.')
        #results['cpts'] = results['cpts'].apply(lambda x: ast.literal_eval(x) if type(x) == str else np.nan)
        Ns=results.drop_duplicates(subset=['identifier']).groupby(['task_name', 'modality'])['duration_labels'].value_counts()
    
    print(f'Number of identifiers: {Ns.to_frame().sum().item()}')
    yellow('Number of different lambda parameter per modality')
    display(
        results.sort_values(["penalty"])
        .groupby(["task_name", "modality"])[
            ["penalty", "participant_id", "identifier", "fps"]
        ]
        .agg(
            {
                "identifier": lambda x: len(np.unique(x)),
                "penalty": lambda x: len(np.unique(x)),
            }
        ).transpose())
        
        
    if aggregate_per_experiment:  # TODO: extend to categorical attribute (e.g. task_name, modality, etc.)
        raise NotImplementedError
        # results = (
        #     results.groupby(["task_name", "modality", "identifier", "penalty", 'video_representation_path'])
        #     .agg("mean")
        #     .reset_index(drop=False)
        # )  # Note: aggregate accross experiment_id
    else:
        pass#results = results.drop_duplicates(["task_name", "modality", "identifier", "penalty"])

    return results

# def parse_args():
#     parser = argparse.ArgumentParser(
#         'Extract TAD features using the videomae model', add_help=False)

#     parser.add_argument(
#         '--cuda',
#         default='0',
#         type=str,
#         help='GPU id to use')

#     parser.add_argument(
#         '--reversed',
#         default=False,
#         action='store_true',
#         help='Whether processing videos in reverse order')

#     return parser.parse_args()


# if __name__ == '__main__':

#     args = parse_args()
#     main()
#     sys.exit(0)
#     sys.exit(0)
