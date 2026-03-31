"""Generate base dataset metadata by scanning the data directory.

When to run: After registration and folder structure initialization.
Prerequisites: Data root directory with task/participant/modality folders.
Outputs: Loads and prints dataset metadata via the base dataset loader.
Usage: python -m smartflat.features.consolidation.main_create_metadata
"""

import os
import sys


from smartflat.datasets.loader import get_dataset


def main():   
    print('Generating dataset...') 
    dset = get_dataset(dataset_name = 'base', scenario='all')

if __name__ == '__main__':
    main()
    sys.exit(0)
    