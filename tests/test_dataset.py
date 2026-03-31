import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


def test_dataset(
    dataset_class,
    root_dir: str,
    num_epochs: int = 5,
    ):

    # Initialize dataset from scratch
    tic = time.time()
    dset = dataset_class(root_dir=root_dir)
    toc = time.time()

    print(
        "Dataset initialisation time: \n"
        + f"  download    : { ((toc - tic) / 60)}2.1f min \n"
        + f"  per 1k rows : {(1e3 * (toc - tic) / len(dset))}2.4f sec \n"
    )

    # training
    dloader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=4)

    # iterate over data loader, batching = 8
    times = []
    for _ in range(num_epochs):
        tic = time.time()
        total = 0.0
        for inputs, _labels, _ids in dloader:
            if isinstance(inputs, torch.Tensor):
                total += inputs.mean()
            else:
                total += inputs[0].mean()

        times += [time.time() - tic]

    print("Times:", times)
    
    
def test_dataset_video_representations(
    dataset_class,
    root_dir: str,
    num_epochs: int = 1,
    ):

    # Initialize dataset from scratch
    tic = time.time()
    dset = dataset_class(root_dir=root_dir)
    toc = time.time()

    print(
        "Dataset initialisation time: \n"
        + f"  download    : { ((toc - tic) / 60)}2.1f min \n"
    )

    # training
    dloader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=4)

    # iterate over data loader, batching = 8
    times = []
    for _ in range(num_epochs):
        tic = time.time()
        total = 0.0
        for inputs, _labels, _ids in dloader:
            
            if not isinstance(inputs, torch.Tensor):
                inputs = inputs[0]
                

            if np.sum(np.isnan(inputs)) > 0:
                print(f"[Warning] {_ids} video embeddings has nan")

        times += [time.time() - tic]

    print("Times:", times)
    print("Total:", total)
