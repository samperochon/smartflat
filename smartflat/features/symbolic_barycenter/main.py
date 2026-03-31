
import os
import sys
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap.umap_ as umap
from aeon.clustering.averaging import elastic_barycenter_average
from aeon.datasets import load_classification
from aeon.distances import (
    shape_dtw_alignment_path,
    shape_dtw_cost_matrix,
    twe_alignment_path,
    twe_cost_matrix,
    twe_pairwise_distance,
)
from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._eshape_dtw import eshape_dtw_cost_matrix
from aeon.distances._rtwe import (
    rtwe_alignment_path,
    rtwe_alignment_path_with_costs,
    rtwe_distance,
)



from smartflat.configs.loader import import_config
from smartflat.features.symbolic_barycenter.visualization import (
    plot_chronogram_alignment,
    plot_distance_violin,
    plot_pairwise_twe_distances_by_group,
    plot_rtwe,
    plot_shapes_signal,
    plot_signals,
)


def get_segments(labels):
    labels = np.array(labels)
    change = np.where(labels[1:] != labels[:-1])[0] + 1
    start_idxs = np.r_[0, change]
    end_idxs = np.r_[change, len(labels)]
    values = labels[start_idxs]
    return list(zip(start_idxs, end_idxs, values))

def _add_inf_to_out_of_bounds_cost_matrix(
    cost_matrix: np.ndarray, bounding_matrix: np.ndarray
) -> np.ndarray:
    x_size, y_size = cost_matrix.shape
    for i in range(x_size):
        for j in range(y_size):
            if not bounding_matrix[i, j]:
                cost_matrix[i, j] = np.inf

    return cost_matrix

def didactic_eshape_dtw_cost_matrix(x, y, nu, lmbda, precomputed_distances=None, itakura_max_slope_eshape_dtw=None, window_eshape_dtw=None, step_sequ=1, lmbda_rtwe = 0.1, nu_rtwe=0.01, window_rtwe = 1, t_max = 3000, title='', cmap='coolwarm', n_plot_max=5, verbose=False):


    exp_title = f'Edit-Shape DTW with rTWE \n{title} step_sequ={step_sequ}\nitakura_max_slope_eshape_dtw={itakura_max_slope_eshape_dtw} and window_eshape_dtw={window_eshape_dtw}\nlmbda_rtwe={lmbda_rtwe} and nu_rtwe={nu_rtwe}'

    if verbose:
        plot_signals(x, y, title='Dyad Chronograms', cmap=cmap, t_max=t_max)


    x_size = x.shape[1]
    y_size = y.shape[1]
    nx = len(range(1, x_size, step_sequ))
    ny = len(range(1, y_size, step_sequ))

    x_cols = [np.ascontiguousarray(x[:, i][None, :]) for i in range(1, x_size, step_sequ)]
    y_cols = [np.ascontiguousarray(y[:, i][None, :]) for i in range(1, y_size, step_sequ)]

    match_same_costs, match_previous_costs, del_x_costs, del_y_costs = [], [], [], []
    total_costs, i_vals, j_vals = [], [], []
    option_paths = []
    x_neg1_ratio, y_neg1_ratio = [], []

    # 1) Edit Shape DTW: Outer-loop over x and y sequences 
    bounding_matrix_eshape_dtw = create_bounding_matrix(nx, ny, window_eshape_dtw, None if nx != ny else itakura_max_slope_eshape_dtw)
    plt.figure(figsize=(10, 4))
    plt.imshow(bounding_matrix_eshape_dtw, cmap='gray', aspect='auto')
    plt.title(f'Bounding Matrix with Window {window_eshape_dtw} and Itakura Max Slope {itakura_max_slope_eshape_dtw}', fontsize=12, fontweight='bold')
    plt.colorbar(label='Bounding Matrix Value')
    plt.xlabel("Time in y", fontsize=14)
    plt.ylabel("Time in x", fontsize=14)
    plt.show()


    cost_matrix = np.zeros((nx, ny)) # +1 for the background row and column
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf
    del_add = nu + lmbda


    t0 = time.time()
    n_iter = 0; n_plot = 0
    for idx, (i, j) in enumerate(((i, j) for i in range(1, nx) for j in range(1, ny))):

        if bounding_matrix_eshape_dtw[i - 1, j - 1]:
            n_iter+=1
            print(f'Processing cost matrix cell: {100 * idx / (nx * ny):.2f} % over total: {(nx * ny)} cells', end='\r')
            # 2) registered Time Warp Edit Distance between state (i,k) and state (j, l): Inner-loop over xi and yj elements
                
            # Get the previous columns
            xip, xi = x_cols[i  - 1], x_cols[i]
            yjp, yj = y_cols[j - 1], y_cols[j]
            
            
            # Deletion in x
            del_x_rtwe_dist, rtwe_cost_matrix, paths, path_costs =  rtwe_alignment_path_with_costs(xip, xi, nu=nu_rtwe, lmbda=lmbda_rtwe, precomputed_distances=precomputed_distances)
            #del_x = cost_matrix[i - 1, j] + del_x_rtwe_dist + del_add
            del_x = cost_matrix[i - 1, j] + del_x_rtwe_dist + del_add
            # # Deletion in y
            del_y_rtwe_dist, rtwe_cost_matrix, paths, path_costs =  rtwe_alignment_path_with_costs(yjp, yj, nu=nu_rtwe, lmbda=lmbda_rtwe, precomputed_distances=precomputed_distances)
            del_y = cost_matrix[i, j - 1] + del_y_rtwe_dist + del_add

            # Match
            match_same_rtwe_d, rtwe_cost_matrix, paths, path_costs =  rtwe_alignment_path_with_costs(xi, yj, nu=nu_rtwe, lmbda=lmbda_rtwe, precomputed_distances=precomputed_distances)
            match_previous_rtwe_d, rtwe_cost_matrix, paths, path_costs =  rtwe_alignment_path_with_costs(xip, yjp, nu=nu_rtwe, lmbda=lmbda_rtwe, precomputed_distances=precomputed_distances)
            match_same_rtwe_d = match_same_rtwe_d / len(path_costs)
            match_previous_rtwe_d = match_previous_rtwe_d / len(path_costs)
            # 
            match = (
                cost_matrix[i - 1, j - 1]
                + match_same_rtwe_d
                + match_previous_rtwe_d
                + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
            )
            #print(del_x, del_y, match)
            cost_matrix[i, j] = min(del_x, del_y, match)
            #print(i, j, cost_matrix[i, j])
            
            
            # # Metrics logging
            option_paths.append(np.argmin([del_x, del_y, match]))
            del_x_costs.append(del_x_rtwe_dist)
            del_y_costs.append(del_y_rtwe_dist)
            match_same_costs.append(match_same_rtwe_d)
            match_previous_costs.append(match_previous_rtwe_d)
            total_costs.append(cost_matrix[i, j])
            i_vals.append(i)
            j_vals.append(j)
            x_neg1_ratio.append(np.sum(xi == -1) / xi.size)
            y_neg1_ratio.append(np.sum(yj == -1) / yj.size) 
            
            
            if verbose or (n_iter % 5000 == 0 and idx > 0 and n_plot < n_plot_max):
                
                n_plot+=1
                dist, rtwe_cost_matrix, paths, path_costs =  rtwe_alignment_path_with_costs(xi, yj, nu=nu, lmbda=lmbda, precomputed_distances=precomputed_distances)
                plot_chronogram_alignment(xi, yj, paths=paths, cost_path=[0] + list(np.ediff1d(path_costs)),  cost_matrix=rtwe_cost_matrix, nu=nu_rtwe, lmbda=lmbda_rtwe, step_sequ=step_sequ, t_max=t_max, window=window_rtwe, precomputed_distances=precomputed_distances, method='rtwe', title=f'\ni={i}, j={j}\nDistance: {dist:.2f}', cmap=cmap, verbose=False)
                
                nu_rtwe_list = [0.001, .5, 2]
                lmbda_rtwe_list = [0.001, 0.01, 0.1, 1]
                window = None  # or set to a float like 0.1

                fig, axes = plt.subplots(len(nu_rtwe_list), len(lmbda_rtwe_list), figsize=(12, 4))
                fig.suptitle(f'rTWE cost matrix for different nu and lmbda values\n{info}', fontsize=16, fontweight='bold')
                for _i, _nu in enumerate(nu_rtwe_list):
                    for _j, _lmbda in enumerate(lmbda_rtwe_list):
                        match_same_rtwe_d, rtwe_cost_matrix, paths, path_costs = rtwe_alignment_path_with_costs(
                            xi, yj, nu=_nu, lmbda=_lmbda, precomputed_distances=precomputed_distances
                        )

                        ax = axes[_i, _j] if len(nu_rtwe_list) > 1 else axes[_j]
                        sns.heatmap(rtwe_cost_matrix, cmap='coolwarm', square=True, ax=ax, cbar=False)
                        ax.set_title(f"nu={_nu}, lmbda={_lmbda}")
                        ax.invert_yaxis()
                        ax.set_xticks([])
                        ax.set_yticks([])

                plt.tight_layout()
                plt.show()
                
        #plot_chronogram_alignment(xip, yjp, nu=nu, t_max=None, lmbda=lmbda, window=window, precomputed_distances=D_G, method='rtwe', title='', cmap=cmap, verbose=False)

    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix_eshape_dtw)
    #cost_matrix = _remove_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)

    final_path = compute_min_return_path(cost_matrix)
    final_path_costs = [cost_matrix[i, j] for (i, j) in final_path]
    final_distance = cost_matrix[nx - 2, ny - 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cost_matrix, origin='lower', cmap='coolwarm', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Alignment Cost")
    path_x, path_y = zip(*final_path)
    ax.plot(path_y, path_x, color='cyan', linewidth=2, label='Alignment Path')
    ax.set_title(f"eShape-DTW distance matrix and alignement path\nFinal Distance: {final_distance:.2f}\n{exp_title}", fontsize=14)
    ax.set_xlabel("Sequence Y Index")
    ax.set_ylabel("Sequence X Index")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    plot_chronogram_alignment(raw_x[:, :t_max],raw_y[:, :t_max], paths=final_path, cost_path=[0] + list(np.ediff1d(final_path_costs)),  
                              cost_matrix=cost_matrix, nu=nu, step_sequ=step_sequ, lmbda=lmbda, window=window, precomputed_distances=D_G, method='rtwe', title=f'\nMatch cost (ediff of cost path): {match_same_rtwe_d:.2f}', cmap=cmap, verbose=False)

    green(f'Total number of iteration to compute the eShapeDTW cost matrix:{n_iter}')
    n_max = 100
    plot_rtwe(match_same_costs[:n_max], match_previous_costs[:n_max], del_x_costs[:n_max], del_y_costs[:n_max], np.ediff1d(total_costs[:n_max+1]), option_paths[:n_max], title=f'rTWE cost matrix (zoom) for{exp_title}')
    plot_rtwe(match_same_costs, match_previous_costs, del_x_costs, del_y_costs, total_costs, option_paths, title=f'RTWE Cost Matrix for {exp_title}')



