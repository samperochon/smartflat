"""Bi-scale (coarse and fine-grained) temporal segmentation of the dataset.

The script performs the following steps:
    1. Runs the change-point-detection experiment to estimate the change-point frequency per penalty values.
    2. Aggregates the results of the experiment and estimates the parameters of the reversed sigmoid function.
    3. Computes the empirical estimate of the two extrema of the third derivative.
    4. Saves the results to a CSV file.
    5. Saves the penalty values for the deployment stage of the change-point-detection.
    6. Runs the deployment experiments.

The script uses various modules and libraries such as ast, multiprocessing, os, sys, time, glob, pprint, typing, matplotlib, numpy, pandas, joblib, sklearn, torch, sympy, and scipy.

Sam Perochon (sam.perochon@ens-paris-saclay.fr)    
July 2024.
TODO: Polish for final packaging ? 
"""

import ast
import multiprocessing
import os
import sys
import time
from glob import glob
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.utils import resample
from torch.utils.data import DataLoader

#from typing import Any, Callable, Dict, Literal, Optional




import sympy as sp
from scipy.optimize import curve_fit

from smartflat.configs.loader import import_config
from smartflat.engine.change_point_detection import get_results_change_point_detection
from smartflat.engine.change_point_detection import main as main_change_point_detection
from smartflat.utils.utils import add_cols_suffixes
from smartflat.utils.utils_io import get_data_root


def main(config_name='ChangePointDetectionAllConfig', experiment_id: str=None, model_params: dict = None, dataset_params: dict = None, continue_process=False, check_results=False):

    # 1) Run the change-point-detection experiment (used for the estimation of f_{cpts} (\lambda) = \frac{L \exp(-k (\log(\lambda) - x0))}{1 + \exp(-k (\log(\lambda) - x0))}

    config_name='ChangePointDetectionExperimentConfig'
    config = import_config(config_name)
    penalty_list = np.logspace(-1, 2, config.model_params['num_samples_penalty'])

    results = get_results_change_point_detection(config, use_stored=True)

    t0 = time.time(); t_init = time.time()
    for task_name in ['cuisine', 'lego']:
        for modality in ['Tobii', 'GoPro1', 'GoPro2', 'GoPro3']:

            for i, penalty in enumerate(penalty_list):
                # print(f'Running {task_name}-{modality}-{i}/{len(penalty_list)} with D={n_pca_components} and whiten={whiten} and penalty={penalty}.')
                main_change_point_detection(config_name=config_name, 
                                            model_params={'penalty': penalty}, 
                                            dataset_params={'task_names': [task_name],
                                                            'modality': [modality], 
                                                            },
                                            results=results,
                                            check_results=True,
                                            continue_process=False,
                                            n_cpus='all')

                print(f'Done experiment {task_name}-{modality}-{i}/{len(penalty_list)}  and penalty={penalty}  in {(time.time() - t0) / 60} min ')
                t0 = time.time()

    print(f'Done change-point detection experiment on the dataset in {(time.time() - t_init) / 60} min ')

    # 2) Get the results and aggregate them for experiments performed multiple times
    config_name='ChangePointDetectionExperimentConfig'
    config = import_config(config_name)
    results_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name); os.makedirs(results_folder, exist_ok=True)
    results = get_results_change_point_detection(config, use_stored=False);print(results.shape)
    results = results.groupby(['task_name', 'modality', 'identifier', 'penalty']).agg('mean').reset_index(drop=False) # Note: aggregate accross experiment_id

    # 3) Estimate  parameters of the reversed sigmoid function (A, L and x_0) and compute empirical estimate of the two extrema of the third derivative.

    # Personalized estimation
    results = append_fit_results(results,  
                                groupby_cols=['task_name', 'modality', #'n_pca_components', 'whiten', 
                                            'identifier'])

    results = append_solved_fitted_curve_results(results,  
                                                groupby_cols=['task_name', 'modality', #'n_pca_components', 'whiten', 
                                                            'identifier'])

    results = add_cols_suffixes(results, 
                            cols=['L_hat', 'k_hat', 'x0_hat', 'L_var', 'k_var', 'x0_var'] + ['lambda_0_log', 'lambda_1_log', 'lambda_0', 'lambda_1'], 
                            suffix='^p')

    # Modality-calibrated estimation
    results = append_fit_results(results,  
                                groupby_cols=['task_name', 'modality', 'n_pca_components', 'whiten'])

    results = append_solved_fitted_curve_results(results,  
                                                groupby_cols=['task_name', 'modality'#,
                                                            #'n_pca_components', 'whiten'
                                                            ])
    results = add_cols_suffixes(results, 
                            cols=['L_hat', 'k_hat', 'x0_hat', 'L_var', 'k_var', 'x0_var'] + ['lambda_0_log', 'lambda_1_log', 'lambda_0', 'lambda_1'], 
                            suffix='^c')

    output_results_path = os.path.join(results_folder, 'results_finals.csv')
    results.to_csv(output_results_path, index=False)
    print(f'Saved dataframe to {output_results_path}.');

    # 5) Save penalty values for the deployment stage of the change-point-detection.

    # Personalized
    config_name = 'ChangePointDetectionDeploymentConfig'
    config = import_config(config_name)

    experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name); os.makedirs(results_folder, exist_ok=True)
    results_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name); os.makedirs(results_folder, exist_ok=True)

    experiment_output_path = os.path.join(experiment_folder, 'lambda_optimal.csv')
    results_output_path = os.path.join(results_folder, 'lambda_optimal.csv')

    # results_final_final.to_csv(experiment_output_path, index=False)
    results.to_csv(results_output_path, index=False)
    print(f'Saved final results (of optimal lambda) in {experiment_output_path} \nand {results_output_path}')

    # Modality-specific (calibrated)
    deployment_config_name = 'ChangePointDetectionCalibratedDeploymentConfig'
    config = import_config(config_name)

    experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name); os.makedirs(results_folder, exist_ok=True)
    results_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name); os.makedirs(results_folder, exist_ok=True)

    experiment_output_path = os.path.join(experiment_folder, 'lambda_optimal.csv')
    results_output_path = os.path.join(results_folder, 'lambda_optimal.csv')

    # results_final_final.to_csv(experiment_output_path, index=False)
    results.to_csv(results_output_path, index=False)

    print(
        f"Saved final results (of optimal lambda) in {experiment_output_path} \nand {results_output_path}"
    )

    # 6) Run the deployment experiments
    from smartflat.engine.change_point_detection import main_deployment

    # Personalized
    deployment_config_name = 'ChangePointDetectionDeploymentConfig'
    main_deployment(config_name, n_cpus='all')

    # Modality-specific (calibrated)
    deployment_config_name = 'ChangePointDetectionCalibratedDeploymentConfig'
    main_deployment(config_name, n_cpus='all')

    print(f'Finished. Estimated optimal penalties for both personalized and calibrated modes,' +
          'and used for computing two-scales temporal segmentation on the dataset')

def mode_or_mean_aggregation(series):
    if isinstance(series.iloc[0], str) or isinstance(series.iloc[0], list) : 
        return series.mode().iloc[0] 
    else:
        return series.mean() 

def sample_groupby(grouped, n_keep=2):

    groupby_col = grouped.keys
    # Select the first two groups
    sample_groups = [group for _, (_, group) in zip(range(n_keep), grouped)]
    
    # Combine the selected groups back into a single DataFrame
    return pd.concat(sample_groups).groupby(groupby_col)

# Fit the shape function f_{cpts}(\lambda) to estimate A, k and x0
def append_fit_results(results, groupby_cols = ['task_name', 'modality', 'identifier'],  train_only=False, debug=False):
    """Fit the shape function f_{cpts}(\lambda) to estimate A, k and x0
    """

    grouped = results.groupby(groupby_cols)

    debug = False
    if debug:
        grouped = sample_groupby(grouped, n_keep=4)

    fit_results = grouped.apply(lambda group: apply_fit_curve(group, train_only=train_only)).reset_index()
    return results.merge(fit_results, on=groupby_cols, how='left')


def apply_fit_curve(group, train_only=True):

    if train_only:
        sgroup = group[group['split'] == 'train']   
        print(f'Fitting curve for train {group.name} with {len(group)} -> {len(sgroup)} samples.')
    else:
        sgroup = group
    x_data = sgroup["log_penalty"]
    y_data = sgroup["cpts_frequency"]
    
    return pd.Series(
        fit_curve(x_data, y_data, verbose=False),
        index=["L_hat", "k_hat", "x0_hat", "L_var", "k_var", "x0_var"],
    )


def fit_curve(x_data, y_data, L = 50, k=1, x0=1, bounds = ((10, 0.05, 0), (2000,  2., 100)), verbose=False):
    """"Estimate the parameters of the change-point frequency perpenalty values.

    Parameters for the sigmoid function
        L = 50  # Maximum value
        k = 1  # Steepness
        x0 = 1  # Midpoint

        bounds = ((35, 0.1, 1.), (40,  2., 3.))
    """

    def f(x, L, k, x0):
        return L*np.exp(-k * (x - x0)) / (1 + np.exp(-k * (x - x0)))

    p0 = [L, k, x0]  

    params, params_covariance = curve_fit(f, x_data, y_data, p0=p0, maxfev=100000, bounds=bounds)
    L_hat, k_hat, x0_hat = params; L_var, k_var, x0_var = np.diag(params_covariance)

    if verbose:
        x_grid = np.linspace(x_data.min(), x_data.max(), 100); y_hat = f(x_grid,  L_hat, k_hat, x0_hat)
        plt.figure(figsize=(10, 2))
        plt.title(f'Empirical samples and estimated curves \nL={L_hat:.2f} (L_v={L_var:.1e})\nk={k_hat:.2f} (k_v={k_var:.1e})\nx0={x0_hat:.2f} (x0_v={x0_var:.1e})')
        plt.scatter(x=x_data, y=y_data,  linewidth=2, label='empirical points')
        plt.plot(x_grid, y_hat,  linewidth=1, linestyle='--', color='red', label='estimated fit')
        plt.ylabel('cpts frequency'); plt.xlabel('log(penaly)')
        plt.legend()
        plt.show()

    return L_hat, k_hat, x0_hat, L_var, k_var, x0_var


# Find extrema of the second derivatives of the shape functions to estimate optimal penalty values (lambda_0 and lambda_1)
def solve_fitted_curve(L_value, k_value, x0_value, verbose=False):
    """Find the two solutions that cancel the third derivatives of the fitted curves"""

    # Define the symbolic variable, function, and third derivative
    x, L, k, x0 = sp.symbols('x L k x0')
    f = L * sp.exp(-k * (x - x0)) / (1 + sp.exp(-k * (x - x0)))
    third_derivative = sp.diff(f, x, x, x)
    critical_points = sp.solve(third_derivative, x)
    critical_points_numerical = [float(cp.subs({L: L_value, k: k_value, x0: x0_value}).evalf()) for cp in critical_points]
    critical_points_numerical_exp = [np.exp(cp) for cp in critical_points_numerical]
    if verbose:
    
        second_derivative = sp.diff(f, x, x)
    
        # Convert the second derivative to a numerical function
        f_func = sp.lambdify((x, L, k, x0), f, 'numpy')
        second_derivative_func = sp.lambdify((x, L, k, x0), second_derivative, 'numpy')
        third_derivative_func = sp.lambdify((x, L, k, x0), third_derivative, 'numpy')
        
        x_values = np.linspace(-10, 10, 1000)
        f_values = f_func(x_values, L_value, k_value, x0_value)
        second_derivative_values = second_derivative_func(x_values, L_value, k_value, x0_value)
        third_derivative_values = third_derivative_func(x_values, L_value, k_value, x0_value)
    
        print(f"Critical point lambda_0 = {critical_points_numerical[0]:.2f} lambda_1= {critical_points_numerical[1]:.2f}")
        
        plt.figure(figsize=(20, 4))
        plt.plot(x_values, f_values, label="f(x)")
        plt.plot(x_values, second_derivative_values, label="f'''(x) second derivative")
        plt.plot(x_values, third_derivative_values, label="f'''(x) third derivative")
        _ = [plt.axvline(cp, color='r', linestyle='--', label=f'$\lambda_{i}$') for i, cp in enumerate(critical_points_numerical)]
        plt.xlabel("x")
        plt.ylabel("")
        plt.title("Curve fitting")
        plt.legend()
        plt.show()
        
    return critical_points_numerical + critical_points_numerical_exp

def append_solved_fitted_curve_results(results, groupby_cols=['task_name', 'modality', 'n_pca_components', 'whiten', 'identifier']):

    agg_results = results.groupby(groupby_cols)[['L_hat', 'k_hat', 'x0_hat']].agg('mean').reset_index()
    fit_results = agg_results.apply(apply_solve_fitted_curve, axis=1)
    agg_results = pd.concat([agg_results[groupby_cols], fit_results], axis=1)
    return results.merge(agg_results, on=groupby_cols, how='left')

def apply_solve_fitted_curve(row):
    
    return pd.Series(solve_fitted_curve(row.L_hat, row.k_hat, row.x0_hat, verbose=False), index=['lambda_0_log', 'lambda_1_log', 'lambda_0', 'lambda_1'])


if __name__ == '__main__':

    config_name = "ChangePointDetectionExperimentHighMinConfig"
    
    for config_name in ['ChangePointDetectionExperimentHighMinConfig', 'ChangePointDetectionExperimentConfig']:
        config = import_config(config_name)
        penalty_list = np.logspace(-1, 2, config.model_params["num_samples_penalty"])
        penalty_list = np.logspace(-1, 2, config.model_params["num_samples_penalty"]) 
        penalty_list = list(np.logspace(-1, 2, config.model_params["num_samples_penalty"])[:-2]) + list(np.linspace(70, 100, 10)[1:-1])
        penalty_list = list(np.logspace(2, 3, 10))[1:]
        #results = get_results_change_point_detection(config, use_stored=False)

        # 1) Run the change-point-detection experiment (used for the estimation of f_{cpts} (\lambda) = \frac{L \exp(-k (\log(\lambda) - x0))}{1 + \exp(-k (\log(\lambda) - x0))}
        t0 = time.time()
        t_init = time.time()
        for task_name in ["cuisine", "lego"]:
            for modality in ["Tobii"]:#, "GoPro1", "GoPro2", "GoPro3"]:

                for i, penalty in enumerate(penalty_list):
                    # print(f'Running {task_name}-{modality}-{i}/{len(penalty_list)} with D={n_pca_components} and whiten={whiten} and penalty={penalty}.')
                                
                    dset = main_change_point_detection(
                        config_name=config_name,
                        model_params={"penalty": penalty},
                        dataset_params={
                            "task_names": [task_name],
                            "modality": [modality],
                        },
                        #results=results,
                        check_results=False,
                        continue_process=False,
                        n_cpus="all",
                    )

                    print(
                        f"Done experiment {task_name}-{modality}-{i}/{len(penalty_list)}  and penalty={penalty}  in {(time.time() - t0) / 60} min "
                    )
                    t0 = time.time()
                
        print(
            f"Done change-point detection experiment on the dataset in {(time.time() - t_init) / 60} min "
        )
        #Save the output to a file
        #with open("/home/sam/output.txt", "w") as f:
        #    f.write(captured.stdout)

        # Found a total of 398 experiments folders in ['/diskA/sam_data/data-features/experiments/change-point-detection-experiment'].
        # Initial length of results:  349046


        print('Done :-)')
    print('Done :-)')

