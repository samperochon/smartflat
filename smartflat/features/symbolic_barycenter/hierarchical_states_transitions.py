
import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import fastcluster
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from IPython.display import display
from tqdm import tqdm

# # from vame.util.cli import get_sessions_from_user_input
# # from vame.visualization.community import draw_tree
# # from vame.schemas.states import save_state, CommunityFunctionSchema
# # from vame.schemas.project import SegmentationAlgorithms
# # from vame.logging.logger import VameLogger
# from vame.analysis.pose_segmentation import get_motif_usage


# logger_config = VameLogger(__name__)
# logger = logger_config.logger


def get_adjacency_matrix(
    labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the adjacency matrix, transition matrix, and temporal matrix.

    Parameters
    ----------
    labels : np.ndarray
        Array of cluster labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing: adjacency matrix, transition matrix, and temporal matrix.
    """
    temp_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    adjacency_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    cntMat = np.zeros((n_clusters))
    steps = len(labels)

    for i in range(n_clusters):
        for k in range(steps - 1):
            idx = labels[k]
            if idx == i:
                idx2 = labels[k + 1]
                if idx == idx2:
                    continue
                else:
                    cntMat[idx2] = cntMat[idx2] + 1
        temp_matrix[i] = cntMat
        cntMat = np.zeros((n_clusters))

    for k in range(steps - 1):
        idx = labels[k]
        idx2 = labels[k + 1]
        if idx == idx2:
            continue
        adjacency_matrix[idx, idx2] = 1
        adjacency_matrix[idx2, idx] = 1

    transition_matrix = get_transition_matrix(temp_matrix)
    return adjacency_matrix, transition_matrix, temp_matrix


def get_transition_matrix(
    adjacency_matrix: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Compute the transition matrix from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix.
    threshold : float, optional
        Threshold for considering transitions. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        Transition matrix.
    """
    row_sum = adjacency_matrix.sum(axis=1)
    transition_matrix = adjacency_matrix / row_sum[:, np.newaxis]
    transition_matrix[transition_matrix <= threshold] = 0
    if np.any(np.isnan(transition_matrix)):
        transition_matrix = np.nan_to_num(transition_matrix)
    return transition_matrix


def get_motif_labels(
    config: dict,
    sessions: List[str],
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get motif labels and motif counts for the entire cohort.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    sessions : List[str]
        List of session names.
    model_name : str
        Model name.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'.

    Returns
    -------
    Tuple [np.ndarray, np.ndarray]
        Tuple with:
            - Array of motif labels (integers) of the entire cohort
            - Array of motif counts of the entire cohort
    """
    # TODO  - this is limiting the number of frames to the minimum number of frames in all files
    # Is this intended behavior? and why?
    shapes = []
    for session in sessions:
        path_to_dir = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            segmentation_algorithm + "-" + str(n_clusters),
            "",
        )
        file_labels = np.load(
            os.path.join(
                path_to_dir,
                str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
            )
        )
        shape = len(file_labels)
        shapes.append(shape)
    shapes = np.array(shapes)

    cohort_motif_labels = []
    for session in sessions:
        path_to_dir = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            segmentation_algorithm + "-" + str(n_clusters),
            "",
        )
        file_labels = np.load(
            os.path.join(
                path_to_dir,
                str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
            )
        )
        cohort_motif_labels.extend(
            file_labels
        )  # add each element to community_label for example [1,2,3] instead of [1, [2,3]] #RENAME TO MOTIF_LABELS
    cohort_motif_labels = np.array(cohort_motif_labels)
    cohort_motif_counts = get_motif_usage(cohort_motif_labels, n_clusters)

    return cohort_motif_labels, cohort_motif_counts


def compute_transition_matrices(
    files: List[str],
    labels: List[np.ndarray],
    n_clusters: int,
) -> List[np.ndarray]:
    """
    Compute transition matrices for given files and labels.

    Parameters
    ----------
    files : List[str]
        List of file paths.
    labels : List[np.ndarray]
        List of label arrays.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    List[np.ndarray]:
        List of transition matrices.
    """
    transition_matrices = []
    for i, file in enumerate(files):
        adj, trans, mat = get_adjacency_matrix(labels[i], n_clusters)
        transition_matrices.append(trans)
    return transition_matrices


def create_cohort_community_bag(
    motif_labels: List[np.ndarray],
    trans_mat_full: np.ndarray,
    cut_tree: Optional[None],
    n_clusters: int,
    segmentation_algorithm: Literal["hmm", "kmeans"],
) -> list:
    """
    Create cohort community bag for given motif labels, transition matrix,
    cut tree, and number of clusters. (markov chain to tree -> community detection)

    Parameters
    ----------
    motif_labels : List[np.ndarray]
        List of motif label arrays.
    trans_mat_full : np.ndarray
        Full transition matrix.
    cut_tree : Optional[None]
        Cut line for tree.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'.

    Returns
    -------
    List
        List of community bags.
    """
    communities_all = []
    #unique_labels, usage_full = np.unique(motif_labels, return_counts=True)
    bins = np.arange(motif_labels.min(), motif_labels.max() + 1)
    usage_full, _ = np.histogram(motif_labels, bins=np.append(bins, bins[-1]+1))
    unique_labels = bins

    labels_usage = dict()
    for la, u in zip(unique_labels, usage_full):
        labels_usage[str(la)] = u / np.sum(usage_full)
        
        #print(f"Motif {la} usage: {u} ({labels_usage[str(la)] * 100:.2f}%)")
    print("Total motifs usage:", np.sum(usage_full), f'usage_full.shape {usage_full.shape}, trans_mat_full.shape {trans_mat_full.shape} n_clusters={n_clusters}')
    T = graph_to_tree(
        motif_usage=usage_full,
        transition_matrix=trans_mat_full,
        n_clusters=n_clusters,
        merge_sel=1,
    )
    # results_dir = os.path.join(
    #     config["project_path"],
    #     "results",
    #     "community_cohort",
    #     segmentation_algorithm + "-" + str(n_clusters),
    # )
    # nx.write_graphml(T, os.path.join(results_dir, "tree.graphml"))
    draw_tree(
        T=T,
        fig_width=n_clusters,
        usage_dict=labels_usage,
        save_to_file=False,
        show_figure=True,
        results_dir=None#results_dir,
    )
    # nx.write_gpickle(T, 'T.gpickle')

    if cut_tree is not None:
        # communities_all = traverse_tree_cutline(T, cutline=cut_tree)
        communities_all = bag_nodes_by_cutline(
            tree=T,
            cutline=cut_tree,
            root="Root",
        )
        print("Communities bag:")
        for ci, comm in enumerate(communities_all):
            print(f"Community {ci}: {comm}")
    else:
        plt.pause(0.5)
        flag_1 = "no"
        while flag_1 == "no":
            cutline = int(input("Where do you want to cut the Tree? 0/1/2/3/..."))
            # community_bag = traverse_tree_cutline(T, cutline=cutline)
            community_bag = bag_nodes_by_cutline(
                tree=T,
                cutline=cutline,
                root="Root",
            )
            print(community_bag)
            flag_2 = input("\nAre all motifs in the list? (yes/no/restart)")
            if flag_2 == "no":
                while flag_2 == "no":
                    add = input("Extend list or add in the end? (ext/end)")
                    if add == "ext":
                        motif_idx = int(input("Which motif number? "))
                        list_idx = int(input("At which position in the list? (pythonic indexing starts at 0) "))
                        community_bag[list_idx].append(motif_idx)
                    if add == "end":
                        motif_idx = int(input("Which motif number? "))
                        community_bag.append([motif_idx])
                        print(community_bag)
                    flag_2 = input("\nAre all motifs in the list? (yes/no/restart)")
            if flag_2 == "restart":
                continue
            if flag_2 == "yes":
                communities_all = community_bag
                flag_1 = "yes"
    return communities_all


def get_cohort_community_labels(
    motif_labels: List[np.ndarray],
    cohort_community_bag: list,
) -> List[np.ndarray]:
    """
    Transform kmeans/hmm parameterized latent vector motifs into communities.
    Get cohort community labels for given labels, and community bags.

    Parameters
    ----------
    labels : List[np.ndarray]
        List of label arrays.
    cohort_community_bag : np.ndarray
        List of community bags. Dimensions: (n_communities, n_clusters_in_community)

    Returns
    -------
    List[np.ndarray]
        List of cohort community labels for each file.
    """
    community_labels_all = []
    num_comm = len(cohort_community_bag)
    community_labels = np.zeros_like(motif_labels)
    for i in range(num_comm):
        clust = np.asarray(cohort_community_bag[i])
        for j in range(len(clust)):
            find_clust = np.where(motif_labels == clust[j])[0]
            community_labels[find_clust] = i
    community_labels_all.append(community_labels)
    return community_labels_all


def save_cohort_community_labels_per_session(
    config: dict,
    sessions: List[str],
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: str,
    cohort_community_bag: list,
) -> None:
    for idx, session in enumerate(sessions):
        path_to_dir = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            segmentation_algorithm + "-" + str(n_clusters),
            "",
        )
        file_labels = np.load(
            os.path.join(
                path_to_dir,
                str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
            )
        )
        community_labels = get_cohort_community_labels(
            motif_labels=file_labels,
            cohort_community_bag=cohort_community_bag,
        )
        if not os.path.exists(os.path.join(path_to_dir, "community")):
            os.mkdir(os.path.join(path_to_dir, "community"))
        np.save(
            os.path.join(
                path_to_dir,
                "community",
                f"cohort_community_label_{session}.npy",
            ),
            np.array(community_labels[0]),
        )


##save_state(model=CommunityFunctionSchema)
def community(
    config: dict,
    cut_tree: Optional[None] = None,
    save_logs: bool = True,
) -> None:
    """
    Perform community analysis.
    Fills in the values in the "community" key of the states.json file.
    Saves results files at:
    - project_name/
        - results/
            - community_cohort/
                - segmentation_algorithm-n_clusters/
                    - cohort_community_bag.npy
                    - cohort_community_label.npy
                    - cohort_segmentation_algorithm_label.npy
                    - cohort_transition_matrix.npy
                    - hierarchy.pkl
            - session_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/
                        - community/
                            - cohort_community_label_session_name.npy

    Parameters
    ----------
    config : dict
        Configuration parameters.
    cut_tree : int, optional
        Cut line for tree. Defaults to None.
    save_logs : bool, optional
        Whether to save logs. Defaults to True.

    Returns
    -------
    None
    """
    if save_logs:
        log_path = Path(config["project_path"]) / "logs" / "community.log"
        logger_config.add_file_handler(str(log_path))

    model_name = config["model_name"]
    n_clusters = config["n_clusters"]
    segmentation_algorithms = config["segmentation_algorithms"]

    # Get sessions
    if config["all_data"] in ["Yes", "yes", "True", "true", True]:
        sessions = config["session_names"]
    else:
        sessions = get_sessions_from_user_input(
            config=config,
            action_message="run community analysis",
        )

    print("---------------------------------------------------------------------")
    print(f"Community analysis for model: {model_name} \n")
    for seg in segmentation_algorithms:
        print(f"Community analysis for segmentation algorithm {seg} with {n_clusters} clusters")
        path_to_dir = Path(
            os.path.join(
                config["project_path"],
                "results",
                "community_cohort",
                seg + "-" + str(n_clusters),
            )
        )
        if not path_to_dir.exists():
            path_to_dir.mkdir(parents=True, exist_ok=True)

        # STEP 1
        cohort_motif_labels, cohort_motif_counts = get_motif_labels(
            config=config,
            sessions=sessions,
            model_name=model_name,
            n_clusters=n_clusters,
            segmentation_algorithm=seg,
        )
        np.save(
            os.path.join(
                path_to_dir,
                "cohort_" + seg + "_label" + ".npy",
            ),
            cohort_motif_labels,
        )
        print(f"Cohort motif labels from {seg} saved")
        np.save(
            os.path.join(
                path_to_dir,
                "cohort_" + seg + "_count" + ".npy",
            ),
            cohort_motif_counts,
        )
        print(f"Cohort motif counts from {seg} saved")
        print(cohort_motif_counts)

        # STEP 2
        _, trans_mat_full, _ = get_adjacency_matrix(
            labels=cohort_motif_labels,
            n_clusters=n_clusters,
        )
        np.save(
            os.path.join(
                path_to_dir,
                "cohort_transition_matrix" + ".npy",
            ),
            trans_mat_full,
        )
        print("Cohort transition matrix saved")

        # STEP 3
        cohort_community_bag = create_cohort_community_bag(
            config=config,
            motif_labels=cohort_motif_labels,
            trans_mat_full=trans_mat_full,
            cut_tree=cut_tree,
            n_clusters=n_clusters,
            segmentation_algorithm=seg,
        )
        # convert cohort_community_bag to dtype object numpy array because it is an inhomogeneous list
        cohort_community_bag = np.array(cohort_community_bag, dtype=object)
        np.save(
            os.path.join(
                path_to_dir,
                "cohort_community_bag" + ".npy",
            ),
            cohort_community_bag,
        )
        print("Community bag saved")

        # STEP 4
        community_labels_all = get_cohort_community_labels(
            motif_labels=cohort_motif_labels,
            cohort_community_bag=cohort_community_bag,
        )
        np.save(
            os.path.join(
                path_to_dir,
                "cohort_community_label" + ".npy",
            ),
            community_labels_all,
        )
        print("Community labels saved")

        with open(os.path.join(path_to_dir, "hierarchy" + ".pkl"), "wb") as fp:  # Pickling
            pickle.dump(cohort_community_bag, fp)

        # Added by Luiz - 11/10/2024
        # Saves the full community labels list for each one of sessions
        # This is useful for further analysis when cohort=True
        save_cohort_community_labels_per_session(
            config=config,
            sessions=sessions,
            model_name=model_name,
            n_clusters=n_clusters,
            segmentation_algorithm=seg,
            cohort_community_bag=cohort_community_bag,
        )
        
        
# pose_segmentation.py


def get_motif_usage(
    session_labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Count motif usage from session label array.

    Parameters
    ----------
    session_labels : np.ndarray
        Array of session labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Array of motif usage counts.
    """
    motif_usage = np.zeros(n_clusters)
    for i in range(n_clusters):
        motif_count = np.sum(session_labels == i)
        motif_usage[i] = motif_count
    # Include warning if any unused motifs are present
    unused_motifs = np.where(motif_usage == 0)[0]
    if unused_motifs.size > 0:
        print(f"Warning: The following motifs are unused: {unused_motifs}")
    return motif_usage



# tree_hierarchy.py


def merge_func(
    transition_matrix: np.ndarray,
    n_clusters: int,
    motif_norm: np.ndarray,
    merge_sel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge nodes in a graph based on a selection criterion.

    Parameters
    ----------
    transition_matrix : np.ndarray
        The transition matrix of the graph.
    n_clusters : int
        The number of clusters.
    motif_norm : np.ndarray
        The normalized motif matrix.
    merge_sel : int
        The merge selection criterion.
        - 0: Merge nodes with highest transition probability.
        - 1: Merge nodes with lowest cost.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the merged nodes.
    """
    if merge_sel == 0:
        # merge nodes with highest transition probability
        cost = np.max(transition_matrix)
        merge_nodes = np.where(cost == transition_matrix)
    elif merge_sel == 1:
        cost_temp = 100
        for i in range(n_clusters):
            for j in range(n_clusters):
                try:
                    cost = motif_norm[i] + motif_norm[j] / np.abs(transition_matrix[i, j] + transition_matrix[j, i])
                except ZeroDivisionError:
                    print(
                        "Error: Transition probabilities between motif "
                        + str(i)
                        + " and motif "
                        + str(j)
                        + " are zero."
                    )
                if cost <= cost_temp:
                    cost_temp = cost
                    merge_nodes = (np.array([i]), np.array([j]))
    else:
        raise ValueError("Invalid merge selection criterion. Please select 0 or 1.")
    return merge_nodes



def graph_to_tree(
    motif_usage: np.ndarray,
    transition_matrix: np.ndarray,
    n_clusters: int,
    merge_sel: int = 1,
) -> nx.Graph:
    """
    Convert a graph to a tree.

    Parameters
    ----------
    motif_usage : np.ndarray
        The motif usage matrix.
    transition_matrix : np.ndarray
        The transition matrix of the graph.
    n_clusters : int
        The number of clusters.
    merge_sel : int, optional
        The merge selection criterion. Defaults to 1.
        - 0: Merge nodes with highest transition probability.
        - 1: Merge nodes with lowest cost.

    Returns
    -------
    nx.Graph
        The tree.
    """
    if merge_sel == 1:
        # motif_usage_temp = np.load(path_to_file+'/behavior_quantification/motif_usage.npy')
        motif_usage_temp = motif_usage
        motif_usage_temp_colsum = motif_usage_temp.sum(axis=0)
        motif_norm = motif_usage_temp / motif_usage_temp_colsum
        motif_norm_temp = motif_norm.copy()
    else:
        motif_norm_temp = None

    merging_nodes = []
    hierarchy_nodes = []
    trans_mat_temp = transition_matrix.copy()
    is_leaf = np.ones((n_clusters), dtype="int")
    node_label = []
    leaf_idx = []

    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1) == 0)
        reduction = len(temp) + 1
    else:
        reduction = 1

    for i in range(n_clusters - reduction):
        nodes = merge_func(
            trans_mat_temp,
            n_clusters,
            motif_norm_temp,
            merge_sel,
        )

        if np.size(nodes) >= 2:
            nodes = np.array([nodes[0][0], nodes[1][0]])

        if is_leaf[nodes[0]] == 1:
            is_leaf[nodes[0]] = 0
            node_label.append("leaf_left_" + str(i))
            leaf_idx.append(1)

        elif is_leaf[nodes[0]] == 0:
            node_label.append("h_" + str(i) + "_" + str(nodes[0]))
            leaf_idx.append(0)

        if is_leaf[nodes[1]] == 1:
            is_leaf[nodes[1]] = 0
            node_label.append("leaf_right_" + str(i))
            hierarchy_nodes.append("h_" + str(i) + "_" + str(nodes[1]))
            leaf_idx.append(1)

        elif is_leaf[nodes[1]] == 0:
            node_label.append("h_" + str(i) + "_" + str(nodes[1]))
            hierarchy_nodes.append("h_" + str(i) + "_" + str(nodes[1]))
            leaf_idx.append(0)

        merging_nodes.append(nodes)

        node1_trans_x = trans_mat_temp[nodes[0], :]
        node2_trans_x = trans_mat_temp[nodes[1], :]

        node1_trans_y = trans_mat_temp[:, nodes[0]]
        node2_trans_y = trans_mat_temp[:, nodes[1]]

        new_node_trans_x = node1_trans_x + node2_trans_x
        new_node_trans_y = node1_trans_y + node2_trans_y

        trans_mat_temp[nodes[1], :] = new_node_trans_x
        trans_mat_temp[:, nodes[1]] = new_node_trans_y

        trans_mat_temp[nodes[0], :] = 0
        trans_mat_temp[:, nodes[0]] = 0

        trans_mat_temp[nodes[1], nodes[1]] = 0

        if merge_sel == 1:
            motif_norm_1 = motif_norm_temp[nodes[0]]
            motif_norm_2 = motif_norm_temp[nodes[1]]
            new_motif = motif_norm_1 + motif_norm_2

            motif_norm_temp[nodes[0]] = 0
            # motif_norm_temp[nodes[1]] = 0
            motif_norm_temp[nodes[1]] = new_motif

    merge = np.array(merging_nodes)

    T = nx.Graph()

    T.add_node("Root")
    node_dict = {}

    if leaf_idx[-1] == 0:
        temp_node = "h_" + str(merge[-1, 1]) + "_" + str(28)
        T.add_edge(temp_node, "Root")
        node_dict[merge[-1, 1]] = temp_node

    if leaf_idx[-1] == 1:
        T.add_edge(merge[-1, 1], "Root")

    if leaf_idx[-2] == 0:
        temp_node = "h_" + str(merge[-1, 0]) + "_" + str(28)
        T.add_edge(temp_node, "Root")
        node_dict[merge[-1, 0]] = temp_node

    if leaf_idx[-2] == 1:
        T.add_edge(merge[-1, 0], "Root")

    idx = len(leaf_idx) - 3

    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1) == 0)
        reduction = len(temp) + 2
    else:
        reduction = 2

    for i in range(n_clusters - reduction)[::-1]:
        if leaf_idx[idx - 1] == 1:
            if merge[i, 1] in node_dict:
                T.add_edge(merge[i, 0], node_dict[merge[i, 1]])
            else:
                T.add_edge(merge[i, 0], temp_node)

        if leaf_idx[idx] == 1:
            if merge[i, 1] in node_dict:
                T.add_edge(merge[i, 1], node_dict[merge[i, 1]])
            else:
                T.add_edge(merge[i, 1], temp_node)

        if leaf_idx[idx] == 0:
            new_node = "h_" + str(merge[i, 1]) + "_" + str(i)
            if merge[i, 1] in node_dict:
                T.add_edge(node_dict[merge[i, 1]], new_node)
            else:
                T.add_edge(temp_node, new_node)

            if leaf_idx[idx - 1] == 1:
                temp_node = new_node
                node_dict[merge[i, 1]] = new_node
            else:
                new_node_2 = "h_" + str(merge[i, 0]) + "_" + str(i)
                T.add_edge(node_dict[merge[i, 1]], new_node_2)
                node_dict[merge[i, 1]] = new_node
                node_dict[merge[i, 0]] = new_node_2

        elif leaf_idx[idx - 1] == 0:
            new_node = "h_" + str(merge[i, 0]) + "_" + str(i)
            if merge[i, 1] in node_dict:
                T.add_edge(node_dict[merge[i, 1]], new_node)
            else:
                T.add_edge(temp_node, new_node)
            node_dict[merge[i, 0]] = new_node

            if leaf_idx[idx] == 1:
                temp_node = new_node
            else:
                new_node = "h_" + str(merge[i, 1]) + "_" + str(i)
                T.add_edge(temp_node, new_node)
                node_dict[merge[i, 1]] = new_node
                temp_node = new_node

        idx -= 2

    return T


def bag_nodes_by_cutline(
    tree: nx.Graph,
    cutline: int = 2,
    root: str = "Root",
):
    """
    Bag nodes of a tree by a cutline.

    Parameters
    ----------
    tree : nx.Graph
        The tree to be bagged.
    cutline : int, optional
        The cutline level. Defaults to 2.
    root : str, optional
        The root node of the tree. Defaults to 'Root'.

    Returns
    -------
    List[List[str]]
        List of bags of nodes.
    """
    if not tree.has_node(root):
        raise ValueError(f"Root node '{root}' not found in the tree.")
    if cutline < 0:
        raise ValueError("Cutline must be a non-negative integer.")

    directed_tree = nx.bfs_tree(tree, source=root)
    leaves = [n for n in directed_tree.nodes() if directed_tree.out_degree(n) == 0]
    bags = {}

    for leaf in leaves:
        path = nx.shortest_path(directed_tree, source=root, target=leaf)
        depth = len(path) - 1
        if depth >= cutline:
            ancestor_at_cutline = path[cutline]
        else:
            ancestor_at_cutline = leaf  # Each leaf in its own bag
        bags.setdefault(ancestor_at_cutline, []).append(leaf)

    return list(bags.values())






# community.py


def hierarchy_pos(
    G: nx.Graph,
    root:Union[str, None] = None,
    width: float = 0.5,
    vert_gap: float = 0.2,
    vert_loc: float = 0,
    xcenter: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """
    Positions nodes in a tree-like layout.
    Ref: From Joel's answer at https://stackoverflow.com/a/29597209/2966723.

    Parameters
    ----------
    G : nx.Graph
        The input graph. Must be a tree.
    root : str, optional
        The root node of the tree. If None, the function selects a root node based on graph type.
        Defaults to None.
    width : float, optional
        The horizontal space assigned to each level. Defaults to 0.5.
    vert_gap : float, optional
        The vertical gap between levels. Defaults to 0.2.
    vert_loc : float, optional
        The vertical location of the root node. Defaults to 0.
    xcenter : float, optional
        The horizontal location of the root node. Defaults to 0.5.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        A dictionary mapping node names to their positions (x, y).
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G,
        root,
        width=1.0,
        vert_gap=0.2,
        vert_loc=0,
        xcenter=0.5,
        pos=None,
        parent=None,
    ):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)



def draw_tree(
    T: nx.Graph,
    fig_width: float = 20.0,
    usage_dict: Dict[str, float] = dict(),
    save_to_file: bool = True,
    show_figure: bool = False,
    results_dir:Union[str, None] = None,
) -> None:
    """
    Draw a tree.

    Parameters
    ----------
    T : nx.Graph
        The tree to be drawn.
    fig_width : int, optional
        The width of the figure. Defaults to 20.
    usage_dict : Dict[str, float], optional
        Dictionary mapping node names to their usage values. Defaults to empty dictionary.
    save_to_file : bool, optional
        Flag indicating whether to save the plot. Defaults to True.
    show_figure : bool, optional
        Flag indicating whether to show the plot. Defaults to False.
    results_dir : str, optional
        The directory to save the plot. Defaults to None.

    Returns
    -------
    None
    """
    # pos = nx.drawing.layout.fruchterman_reingold_layout(T)
    pos = hierarchy_pos(
        G=T,
        root="Root",
        width=10.0,
        vert_gap=0.1,
        vert_loc=0,
        xcenter=50,
    )
    # Nodes appearances
    # Nodes sizes are mapped to a scale between 100 and 61prin00, depending on the usage of the node
    node_labels = dict()
    node_sizes = []
    node_colors = []
    for k in list(T.nodes):
        if isinstance(k, str):
            node_labels[k] = ""
            node_sizes.append(50)
            node_colors.append("#000000")
        else:
            node_labels[k] = str(k)
            size = usage_dict.get(str(k), 0.5)
            node_sizes.append(100 + size * 6000)
            node_colors.append("#46a7e8")

    fig_width = min(max(fig_width, 10.0), 30.0)
    fig = plt.figure(
        num=2,
        figsize=(fig_width, 20.0),
    )
    nx.draw_networkx(
        G=T,
        pos=pos,
        with_labels=True,
        labels=node_labels,
        node_size=node_sizes,
        node_color=node_colors,
    )
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    if save_to_file and results_dir:
        save_fig_path = Path(results_dir) / "tree.png"
        save_fig_pdf_path = Path(results_dir) / "tree.pdf"
        plt.savefig(save_fig_path, bbox_inches="tight")
        plt.savefig(save_fig_pdf_path, bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def visualize_hierarchical_tree(
    config: dict,
    segmentation_algorithm: Literal["hmm", "kmeans"],
) -> None:
    """
    Visualizes the hierarchical tree.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    segmentation_algorithm : Literal["hmm", "kmeans"]
        Segmentation algorithm.

    Returns
    -------
    None
    """
    n_clusters = config["n_clusters"]
    fig_path = (
        Path(config["project_path"])
        / "results"
        / "community_cohort"
        / f"{segmentation_algorithm}-{n_clusters}"
        / "tree.png"
    )
    if not fig_path.exists():
        raise FileNotFoundError(f"Tree figure not found at {fig_path}.")
    img = plt.imread(fig_path)
    plt.figure(figsize=(n_clusters, n_clusters))
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()