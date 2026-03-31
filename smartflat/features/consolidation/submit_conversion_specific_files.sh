#!/bin/bash
# Source this script to set input video path and run ffmpeg

# Set input video paths
input_video="/gpfs/workdir/perochons/data-gold-final/lego/L154_P113_RAYVia_03052023/GoPro2/merged_video.mp4"
input_video_downsized="/gpfs/workdir/perochons/data-gold-final/lego/L154_P113_RAYVia_03052023/GoPro2/merged_video_downsized.mp4"

# Run ffmpeg command on the downsized video
echo "Processing video with ffmpeg..."
ffmpeg -i /gpfs/workdir/perochons/data-gold-final/lego/L154_P113_RAYVia_03052023/GoPro2/merged_video.mp4 -vf fps=25 -c:a copy /gpfs/workdir/perochons/data-gold-final/lego/L154_P113_RAYVia_03052023/GoPro2/merged_video_downsized.mp4

echo "Video processing complete."