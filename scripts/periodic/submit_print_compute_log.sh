#!/bin/bash

#echo 'export DATAROOT=/diskA/sam_data/data' >> ~/.bashrc
#echo 'export DATAROOT=/gpfs/workdir/perochons/data' >> ~/.bashrc
file_path= $DATAROOT/log/skeleton_estimation_computation.log
num_lines=1

tail -n $num_lines $file_path


file_path=$DATAROOT/log/hand_landmarks_representations_computation.log
num_lines=1

tail -n $num_lines $file_path

file_path=$DATAROOT/log/audio_representations_computation.log
num_lines=1

tail -n $num_lines $file_path

file_path=$DATAROOT/log/video_representations_computation.log
num_lines=1

tail -n $num_lines $file_path





