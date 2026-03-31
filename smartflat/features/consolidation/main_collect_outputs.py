"""Collect and aggregate feature extraction outputs from multiple modalities.

When to run: After feature extraction pipelines (video, speech, hands, skeleton) complete.
Prerequisites: Dataset with computed features (video_representation, speech, hand/skeleton landmarks).
Outputs: Aggregated embeddings organized by output type in the data root.
Usage: python -m smartflat.features.consolidation.main_collect_outputs
"""

import argparse
import os
import sys



from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import get_data_root, collect_embeddings


def main(process=False):
    
    dset = get_dataset(dataset_name="base", root_dir=get_data_root('light'), scenario="all")
    # dump_location = os.path.join(get_data_root(), 'dump', 'dump_{}'.format(socket.gethostname())); os.makedirs(dump_location, exist_ok=True)

    for output_type in [
        "video_representation",
        "speech_recognition",
        "speech_representation",
        "hand_landmarks",
        "skeleton_landmarks",
    ]:  # , 'video_representation',]:
        collect_embeddings(
            dset.metadata, output_type=output_type, output_root=get_data_root('light'), process=process, command_output="cp")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect feature extraction outputs.")
    parser.add_argument(
        "-p",
        "--process",
        action="store_true",
        help="Set this flag to enable processing",
    )
    args = parser.parse_args()

    main(process=args.process)
    sys.exit(0)
