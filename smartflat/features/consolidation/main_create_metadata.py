import os
import sys


from smartflat.datasets.loader import get_dataset


def main():   
    print('Generating dataset...') 
    dset = get_dataset(dataset_name = 'base', scenario='all')

if __name__ == '__main__':
    main()
    sys.exit(0)
    