import os
import matplotlib.pyplot as plt
import copy
import numpy as np
import json
import pandas as pd
from collections import defaultdict
from filelock import FileLock

#functions follow same structure as for inital plots (InitialPlots.py). 
#the individual subfunction/ how it works is identical, so I dont want to comment it twice.
def process1_TwoSame_Handedness_InFrames(modified_dataHandLandmark, new_folder_path):
    copied_dataHandLandmark4 = modified_dataHandLandmark

    frame_indices = []
    colors = []
    y_values = []

    for frame_index, frame in enumerate(copied_dataHandLandmark4, start=1):
        hand_counts = {'left': 0, 'right': 0}

        if 'handedness' in frame:
            for hand in frame['handedness']:
                try:
                    hand_type = hand.get('display_name', 'unknown').lower()
                    if 'left' in hand_type:
                        hand_counts['left'] += 1
                    elif 'right' in hand_type:
                        hand_counts['right'] += 1
                except (IndexError, KeyError) as e:
                    continue

            if hand_counts['left'] > 1:
                frame_indices.append(frame_index % 1000)
                colors.append('yellow')  # Multiple left hands
            elif hand_counts['right'] > 1:
                frame_indices.append(frame_index % 1000)
                colors.append('red')  # Multiple right hands
            else:
                frame_indices.append(frame_index % 1000)
                colors.append('darkblue')  # No multiple hands or empty
        else:
            frame_indices.append(frame_index % 1000)
            colors.append('darkblue')  # Empty frame
        y_values.append((frame_index - 1) // 1000 + 1)

    plt.figure(figsize=(15, 5))
    plt.scatter(frame_indices, y_values, c=colors, marker='|')
    plt.yticks(range(1, (max(y_values) + 1)))
    plt.xlabel('Frame Index')
    plt.ylabel('1000 Frame Groups')
    plt.title('Plot of the modified data with averaged score values and handedness')
    
    handles = [
        plt.Line2D([0], [0], marker='|', color='yellow', markersize=10, label='Multiple Left Hands'),
        plt.Line2D([0], [0], marker='|', color='red', markersize=10, label='Multiple Right Hands'),
        plt.Line2D([0], [0], marker='|', color='darkblue', markersize=10, label='No Multiple Hands or Empty')
    ]
    plt.legend(handles=handles, loc='upper right', facecolor='white', edgecolor='black', framealpha=1)

    extra_name = "Plot of data with averaged score values and handedness"
    fig_filename = os.path.join(new_folder_path, f"plot_{extra_name}.png")
    plt.savefig(fig_filename)
    plt.show()
    
def process1_TwoSame_Handedness_InSeconds(modified_dataHandLandmark, new_folder_path, fps):
    copied_dataHandLandmark4 = modified_dataHandLandmark

    frame_indices = []
    colors = []
    y_values = []

    for frame_index, frame in enumerate(copied_dataHandLandmark4, start=1):
        hand_counts = {'left': 0, 'right': 0}

        if 'handedness' in frame:
            for hand in frame['handedness']:
                try:
                    hand_type = hand.get('display_name', 'unknown').lower()
                    if 'left' in hand_type:
                        hand_counts['left'] += 1
                    elif 'right' in hand_type:
                        hand_counts['right'] += 1
                except (IndexError, KeyError) as e:
                    continue

            if hand_counts['left'] > 1:
                frame_indices.append(frame_index % 1000 / fps)
                colors.append('yellow')  # Multiple left hands
            elif hand_counts['right'] > 1:
                frame_indices.append(frame_index % 1000 / fps)
                colors.append('red')  # Multiple right hands
            else:
                frame_indices.append(frame_index % 1000 / fps)
                colors.append('darkblue')  # No multiple hands or empty
        else:
            frame_indices.append(frame_index % 1000 / fps)
            colors.append('darkblue')  # Empty frame
        y_values.append((frame_index - 1) // 1000 + 1)

    plt.figure(figsize=(15, 5))
    plt.scatter(frame_indices, y_values, c=colors, marker='|')
    plt.yticks(range(1, (max(y_values) + 1)))
    plt.xlabel('Time (seconds)')
    plt.ylabel('1000 Frame Groups')
    plt.title('Plot of the modified data with averaged score values and handedness')
    
    handles = [
        plt.Line2D([0], [0], marker='|', color='yellow', markersize=10, label='Multiple Left Hands'),
        plt.Line2D([0], [0], marker='|', color='red', markersize=10, label='Multiple Right Hands'),
        plt.Line2D([0], [0], marker='|', color='darkblue', markersize=10, label='No Multiple Hands or Empty')
    ]
    plt.legend(handles=handles, loc='upper right', facecolor='white', edgecolor='black', framealpha=1)

    extra_name = "Plot of data with averaged score values and handedness in [s]"
    fig_filename = os.path.join(new_folder_path, f"plot_{extra_name}.png")
    plt.savefig(fig_filename)
    plt.show()
    
def process2_TwoSame_Handedness_InFrames(copied_dataHandLandmark, new_folder_path):
    frame_indices = []
    colors = []
    y_values = []

    for frame_index, frame in enumerate(copied_dataHandLandmark, start=1):
        hand_counts = {'left': 0, 'right': 0}
        valid_frame = True

        if 'handedness' in frame:
            for hand in frame['handedness']:
                try:
                    hand_type = hand.get('display_name', 'unknown').lower()
                    if 'left' in hand_type:
                        hand_counts['left'] += 1
                    elif 'right' in hand_type:
                        hand_counts['right'] += 1
                    if hand.get('score', 0) < 0.9:
                        valid_frame = False
                except (IndexError, KeyError) as e:
                    continue

            if not valid_frame:
                frame_indices.append(frame_index % 1000)
                colors.append('darkblue')
            elif hand_counts['left'] > 1:
                frame_indices.append(frame_index % 1000)
                colors.append('yellow')
            elif hand_counts['right'] > 1:
                frame_indices.append(frame_index % 1000)
                colors.append('red')
            else:
                frame_indices.append(frame_index % 1000)
                colors.append('darkblue')
        else:
            frame_indices.append(frame_index % 1000)
            colors.append('darkblue')
        y_values.append((frame_index - 1) // 1000 + 1)

    plt.figure(figsize=(15, 5))
    plt.scatter(frame_indices, y_values, c=colors, marker='|')
    plt.yticks(range(1, (max(y_values) + 1)))
    plt.xlabel('Frame Index')
    plt.ylabel('1000 Frame Groups')
    plt.title('Initial Plot using only modified data with score over 0.9')
    
    handles = [
        plt.Line2D([0], [0], marker='|', color='yellow', markersize=10, label='Multiple Left Hands'),
        plt.Line2D([0], [0], marker='|', color='red', markersize=10, label='Multiple Right Hands'),
        plt.Line2D([0], [0], marker='|', color='darkblue', markersize=10, label='No Multiple Hands or Empty')
    ]
    plt.legend(handles=handles, loc='upper right', facecolor='white', edgecolor='black', framealpha=1)

    extra_name = "Initial Plot of modified data using only scores below 0.9"
    fig_filename = os.path.join(new_folder_path, f"plot_{extra_name}.png")
    plt.savefig(fig_filename)
    plt.show()

def process2_TwoSame_Handedness_InSeconds(copied_dataHandLandmark, new_folder_path, fps):
    frame_indices = []
    colors = []
    y_values = []

    for frame_index, frame in enumerate(copied_dataHandLandmark, start=1):
        hand_counts = {'left': 0, 'right': 0}
        valid_frame = True

        if 'handedness' in frame:
            for hand in frame['handedness']:
                try:
                    hand_type = hand.get('display_name', 'unknown').lower()
                    if 'left' in hand_type:
                        hand_counts['left'] += 1
                    elif 'right' in hand_type:
                        hand_counts['right'] += 1
                    if hand.get('score', 0) < 0.9:
                        valid_frame = False
                except (IndexError, KeyError) as e:
                    continue

            if not valid_frame:
                frame_indices.append(frame_index % 1000 / fps)
                colors.append('darkblue')
            elif hand_counts['left'] > 1:
                frame_indices.append(frame_index % 1000 / fps)
                colors.append('yellow')
            elif hand_counts['right'] > 1:
                frame_indices.append(frame_index % 1000 / fps)
                colors.append('red')
            else:
                frame_indices.append(frame_index % 1000 / fps)
                colors.append('darkblue')
        else:
            frame_indices.append(frame_index % 1000 / fps)
            colors.append('darkblue')
        y_values.append((frame_index - 1) // 1000 + 1)

    plt.figure(figsize=(15, 5))
    plt.scatter(frame_indices, y_values, c=colors, marker='|')
    plt.yticks(range(1, (max(y_values) + 1)))
    plt.xlabel('Time (seconds)')
    plt.ylabel('1000 Frame Groups')
    plt.title('Plot using only modified data with score over 0.9')
    
    handles = [
        plt.Line2D([0], [0], marker='|', color='yellow', markersize=10, label='Multiple Left Hands'),
        plt.Line2D([0], [0], marker='|', color='red', markersize=10, label='Multiple Right Hands'),
        plt.Line2D([0], [0], marker='|', color='darkblue', markersize=10, label='No Multiple Hands or Empty')
    ]
    plt.legend(handles=handles, loc='upper right', facecolor='white', edgecolor='black', framealpha=1)

    extra_name = "Plot of modified data using only scores below 0.9 in [s]"
    fig_filename = os.path.join(new_folder_path, f"plot_{extra_name}.png")
    plt.savefig(fig_filename)
    plt.show()

