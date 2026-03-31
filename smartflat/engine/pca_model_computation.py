"""Perform pca models computation on all available video block representation, per modality and task"""
import multiprocessing
import os
import pickle
import sys
import time
from pprint import pprint
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np


from sklearn.decomposition import PCA

from smartflat.configs import BaseSmartflatConfig
from smartflat.configs.loader import import_config
from smartflat.constants import available_modality, available_tasks
from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import get_api_root, get_data_root


def main():
    # Configs of the experiment

    output_dir = os.path.join(get_api_root(), 'api', 'models', 'artifacts')
    config = import_config('ClusteringPCAConfig')
    t0 = time.time()
    for task_name in available_tasks:

        for modality in available_modality:
            
            for whiten in [False]:
                
                # Modulable 
                if task_name != 'cuisine':
                    pass#continue
                
                if modality != 'Tobii':
                    pass#continue
                                
                config.dataset_params['task_names'] = [task_name]
                config.dataset_params['modality'] = [modality]
                config.do_whiten = whiten

                dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)

                X = np.vstack([dset[i][0] for i in range(len(dset))]) 
                
                n = X.shape[0]
                # Remove nans
                nan_rows = np.any(np.isnan(X), axis=1)
                X = X[~nan_rows]
                print(f'Training data shape for computing the PCA: {X.shape} ({n - X.shape[0]} NAN vectors removed)')

                for n_components in   [10, 25, 50, 75, 100]:
                    
                    if config.do_whiten:
                        filename = f'pca_{task_name}_{modality}_{n_components}_w.pkl'
                    else:
                        filename = f'pca_{task_name}_{modality}_{n_components}.pkl'
                        
                    output_filepath = os.path.join(output_dir, filename)
                    if os.path.isfile(output_filepath):
                        print(f'File {output_filepath} already exists.')
                        #continue

                    print(f'Fitting PCA for {task_name}-{modality}-D={n_components} and whiten={whiten} and {X.shape[0]} training samples.')
                    pca = PCA(n_components=n_components, whiten=config.do_batch_whiten)
                    pca.fit(X)
                    
                    with open(output_filepath, 'wb') as f:
                        pickle.dump(pca, f)
                    print(f'Saved {output_filepath} PCA model. Compute time: {(time.time() - t0)/60} min'); t0=time.time()
                    
if __name__ == '__main__':
    
    main()
    sys.exit(0)