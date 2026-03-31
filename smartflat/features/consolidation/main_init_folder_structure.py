"""Initialize the dataset directory structure from metadata.

When to run: After metadata CSV is created, before feature extraction.
Prerequisites: Path to metadata CSV with task_name, participant_id, and modality columns.
Outputs: Creates nested folder hierarchy (task/participant/modality) under the data root.
Usage: python -m smartflat.features.consolidation.main_init_folder_structure
"""

import argparse
import logging
import os
import subprocess
import sys

import numpy as np
from IPython.display import display

,
)
from smartflat.utils.utils_io import (
    get_data_root,
)

from smartflat.features.consolidation.main_housekeeping import init_dataset_structure


def main(
    path_metadata
):

    init_dataset_structure(path_metadata)




def parse_args():

    parser = argparse.ArgumentParser(description="Init dataset structure ")
    parser.add_argument("-r", "--path_metadata", help="Path to dataset metadata")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    
    if args.path_metadata is None:
        args.path_metadata = os.path.join(get_data_root(), "dataframes", "persistent_metadata", "feature_extraction_dataset_df.csv")
    main(path_metadata=args.path_metadata)
    print("Done")
    sys.exit(0)

