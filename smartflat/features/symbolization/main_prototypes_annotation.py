"""Interactive annotation of video prototypes using pigeon-jupyter.

Supports the recursive prototyping pipeline (Ch. 5): after clustering,
prototypes are manually annotated into 5 categories (Noise, task-definitive,
task-ambiguous, exo-definitive, exo-ambiguous) via an interactive widget.

Requires:
    - pigeon-jupyter: ``pip install pigeon-jupyter``
    - IPython/Jupyter environment for the interactive widget

Usage:
    Called from notebooks during the iterative clustering-annotation loop.
    See ``smartflat/features/symbolization/`` for the full pipeline.
"""

import os
import re
import sys
from glob import glob

import pandas as pd

try:
    from IPython.display import Image, display
except ImportError:
    Image = None
    display = print

from pigeon import annotate

from smartflat.configs.loader import import_config
from smartflat.utils.utils_coding import green
from smartflat.utils.utils_io import get_data_root


def main(config, task_name, modality, n_clusters_folder, permutations=None, overwrite=False, dry=True, local=None):
    """Annotate prototypes for a given task, modality and number of clusters."""
    
    input_image_folder = os.path.join(get_data_root(local=local), 'outputs', config.experiment_name,'figures-clusters', task_name, modality, n_clusters_folder, 'closest')
    print(f'Annotating prototypes from {input_image_folder}')
    output_path = os.path.join(get_data_root(), 'dataframes', 'annotations', 'pigeon-annotations', config.experiment_name, f'prototypes_{n_clusters_folder}.csv'); os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.isfile(output_path) and not dry:
        print(f'Loading annotations from {output_path}')
        annotations_done = pd.read_csv(output_path).image_annotation_path.tolist()
        print(f'Already annotated {len(annotations_done)} images')
    else:
        annotations_done = []

    # 1) Fetch prototypes annotations image paths 
    impaths = [impath for impath in glob(os.path.join(input_image_folder, '*')) if impath not in annotations_done]
    print(f'Remaining {len(impaths)} images to annotate')
    if permutations is not None:
        impaths = permute_paths_and_exlude_missing(impaths, permutations)

    annotations = annotate(
    impaths,
    options=['Noise', 'task-definitive', 'task-ambiguous', 'exo-definitive', 'exo-ambiguous'],
    display_fn=lambda filename: display(Image(filename))
    )

    # 2) Parse annotations and save the results
    res = []
    for image_annotation_path, cluster_type in annotations:
        print(f'Annotated {image_annotation_path} as {cluster_type}')
        
        path = os.path.basename(image_annotation_path)
        n_cluster = parse_prototypes_path(path) 
        
        res.append({'image_annotation_path': image_annotation_path, 
                    'path': path, 
                    'n_cluster': n_cluster,
                    'cluster_type': cluster_type,
                    'cluster_folder': n_clusters_folder
                    })
        
    clusters_annotations_df = pd.DataFrame(res)
    
    print(f'Annotated {len(clusters_annotations_df)} images')

    if os.path.isfile(output_path) and overwrite and not dry:
        
        print(f'Concatenating annotations from {output_path}')
        # Load saved dataframe and concatenate the results
        clusters_annotations_df = pd.concat([pd.read_csv(output_path), clusters_annotations_df], axis=0)
        print(f'Added {len(clusters_annotations_df) - len(pd.read_csv(output_path))} new annotations to {output_path}')
        clusters_annotations_df.to_csv(output_path, index=False)
        green(f'Saved to {output_path}')
        
    elif (len(clusters_annotations_df) > 0) and not dry:
        
        print(f'Concatenating annotations from {output_path}')
        clusters_annotations_df.to_csv(output_path, index=False)
        green(f'Saved to {output_path}')
        
    return 
        


def parse_prototypes_path(path):
    match = re.search(r'_K_(\d+)_', path)
    return int(match.group(1)) if match else None


def permute_paths_and_exlude_missing(paths, permutations):
    order_map = {k: i for i, k in enumerate(permutations)}
    filtered_paths = [p for p in paths if parse_prototypes_path(p) in order_map]
    return sorted(filtered_paths, key=lambda p: order_map[parse_prototypes_path(p)])

if __name__ == '__main__':
    
    
    # LOG ANNOTATION

    #1 - 27/01/2025
    task_name = 'cuisine'
    modality='Tobii'

    n_clusters_folder = 'K_250'

    clustering_config_name='ClusteringDeploymentKmeansConfig'
    config = import_config(clustering_config_name)
    
    # BUG: missing config argument — should be main(config, task_name, modality, n_clusters_folder)
    main(task_name, modality, n_clusters_folder)
    
    #/Users/samperochon/github-repositories/smartflat/data/data-gold-final/dataframes/annotations/pigeon-annotations/clustering-deployment-kmeans-all/prototypes_K_250.csv
    #/Users/samperochon/github-repositories/smartflat/data/data-gold-final/dataframes/annotations/pigeon-annotations/clustering-deployment-kmeans-all/prototypes_K_250_backup.csv
    

