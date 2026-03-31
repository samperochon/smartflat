import os
import sys
import argparse


from smartflat.utils.utils_io import get_data_root, get_host_name
from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_visualization import print_get_last_modified_date
from smartflat.features.consolidation.main_housekeeping import send_hardrive_to_remote


def main(args):  
    

    metadata_local_path = os.path.join(
        get_data_root(), "dataframes", "persistent_metadata", "{}_dataset_df.csv".format(get_host_name())
    )
    metadata_remote_path = os.path.join(
        get_data_root(), "dataframes", "persistent_metadata", "{}_dataset_df.csv".format(args.remote_name)
    )
    path_df_external = None  # os.path.join(get_data_path(), 'dataframes', 'persistent_metadata', 'cheetah_dataset_df.csv')

    print_get_last_modified_date(metadata_local_path)
    print_get_last_modified_date(metadata_remote_path)
    # path_metadata  = os.path.join(get_data_root('smartflat'), 'dataframes', 'persistent_metadata', 'metadata.csv')

    commands = send_hardrive_to_remote(
        metadata_remote_path=metadata_remote_path,
        metadata_local_path=metadata_local_path,
        path_df_external=path_df_external,
        remote_name=args.remote_name,
        is_gold=True,
        process=args.process,
    )
    
    if not args.process:
        print('Turn process on to process: us.send_hardrive_to_remote(process=True)')
        print('Commands:', commands)
        print(f'Found {len(commands)} commands')
    
    
def parse_args():
    
    parser = argparse.ArgumentParser(description='Collate dataset')
    parser.add_argument('-p', '--process', action="store_true", default=False, help='Process the videos')
    parser.add_argument('-r', '--remote_name', default='ruche', help='cheetah or ruche')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    main(args)
    sys.exit(0)
    