#the way this code works is explained in my report for my erasmus internship. 
#the code is structured in different parts, as I merged it in order to put it into Github:
#I put comments in this code, as Part 1-3, as I structured them similarly in my report.
#the first part is about the tracking of the hand landmarks, the second has the goal of removing data with low confidence scores.
#the third is about using a join and discard function on the data.
#The reasons, why we did it like this, is explained in the report.
#If you need the report and it is not available for you, you can always contact me here: alex.s.kreibich@gmail.com

import os
import sys
import time
import matplotlib.pyplot as plt
import copy
import logging
import numpy as np
import json
import pandas as pd
from collections import defaultdict
from filelock import FileLock



# Own tools import
from smartflat.datasets.loader import get_dataset  # noqa: E402
from smartflat.configs.smartflat_config import BaseSmartflatConfig
from smartflat.utils.utils_io import (  # noqa: E402
    fetch_output_path,
    get_api_root,
    get_host_name,
    get_data_root,
    parse_flag,
    parse_identifier,
    parse_path
)

log_dir = os.path.join(get_data_root(), 'log');  os.makedirs(log_dir, exist_ok=True)
                    
logging.basicConfig(filename=os.path.join(log_dir, 'tracking_hand_landmarks_computation.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logging.info("Hands landmarks processing computation")
logger = logging.getLogger('main')


# Own tools import
from smartflat.datasets.loader import get_dataset  # noqa: E402
from smartflat.configs.smartflat_config import BaseSmartflatConfig
from smartflat.utils.utils_io import (  # noqa: E402
    fetch_output_path,
    get_api_root,
    get_host_name,
    get_data_root,
    parse_flag,
    parse_identifier,
    parse_path
)

log_dir = os.path.join(get_data_root(), 'log');  os.makedirs(log_dir, exist_ok=True)
                    
logging.basicConfig(filename=os.path.join(log_dir, 'tracking_hand_landmarks_computation.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logging.info("Hands landmarks processing computation")
logger = logging.getLogger('main')


def main(root_dir=None, overwrite=False):
    
    if root_dir is None:
        root_dir = get_data_root()
        
                
    # -------- 
    # Define model and video dataset path
    model_name = 'tracking_hand_landmarks_v1'
    metrics_path = os.path.join(root_dir, 'dataframes', f'{get_host_name()}_compute_time_tracking_hand_landmarks.csv')
    
    # import operation config
    config = BaseSmartflatConfig()

    # Get video to process
    dset = get_dataset(dataset_name='tracking_hand_landmarks', root_dir=root_dir, scenario='gold_unprocessed_tracking')
    input_paths = dset.metadata.sort_values('size', ascending=True)['hand_landmarks_path'].tolist()
    
    print('Processing:')
    print('\n'.join(input_paths))
    
    metrics = [] 
    #for j, input_path in enumerate(input_paths[:10]): #TODO remove :1 
    for j, row in dset.metadata.dropna(subset=['fps']).iterrows():

        input_path = row.hand_landmarks_path
        fps = row.fps
         
        start_time = time.time()
        logger.info("Processing hand landmarks {}/{} for {}".format(j+1, len(input_paths), input_path))
        
        
        # 1) Initialization of the loop 
        output_path = fetch_output_path(input_path, model_name)
        video_name, _, _, _ = parse_path(input_path)

        print(f'Saving in {output_path}')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print('hello we are here')
        
        
        if os.path.isfile(output_path) and not overwrite:
            logger.info(f"Computation aready {output_path} (overwrite={overwrite})")
            continue
        print('hello we are here 2')
        try:
            print(f"Attempting to open file: {input_path}")
            with open(input_path, 'r') as f:
                raw_data = f.read()
                print("File read successfully, attempting to parse JSON")
                data = json.loads(raw_data)
                print(f"file was opened and parsed successfully: {input_path}")
        except MemoryError:
            logger.error(f"MemoryError: File {input_path} is too large, skipping.")
            continue
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in file {input_path}: {str(e)}, skipping this file...")
            continue
        except Exception as e:
            logger.error(f"Unexpected error when opening file {input_path}: {str(e)}, skipping this file...")
            continue

        print('hello we are here 3')

        deltaTimeJoin = 0.5 #seconds
        deltaTimeDiscard = 0.5 #seconds

        rowNumber = dset.metadata.loc[dset.metadata['hand_landmarks_path'] == input_path]
        #print(rowNumber)
        print('hello we are here 4')
        print(rowNumber)

        print(rowNumber['fps'])

        if not rowNumber.empty:
            fps = rowNumber['fps'].values[0]  # Get the fps value from the first matching row
            print('fps works')

        else:
            #print(f"No matching fps found for {input_paths}")
            continue
        print('hello we are here 5')
        
        join_len = deltaTimeJoin*fps
        discard_len = deltaTimeDiscard*fps


            


        
        # Strat of the processing


        print('the step 1 begins')
        # Step 1: Initial processing using DetectionAlgFin
        modified_data = DetectionAlgFin(data)

 

        # Step 2: Apply join and discard operation
        final_data, one_right_hand_indices, one_left_hand_indices, right_hand_indices, left_hand_indices = join_and_discard_handedness(modified_data, join_len, discard_len)

        # Process the segments
        lists1Right = find_connected_subseriesStartAndEnd(one_right_hand_indices)
        lists1Left = find_connected_subseriesStartAndEnd(one_left_hand_indices)
        lists2Right = find_connected_subseriesStartAndEnd(right_hand_indices)
        lists2Left = find_connected_subseriesStartAndEnd(left_hand_indices)


        lists1Left = loopFindScoreOneLeft(lists1Left, modified_data)
        lists1Right = loopFindScoreOneRight(lists1Right, modified_data)
        lists2Left = loopFindScoreTwoLeft(lists2Left, modified_data)    # Updated function
        lists2Right = loopFindScoreTwoRight(lists2Right, modified_data)  # Updated function


        # Now 'lists2Left' and 'lists2Right' contain only segments with exactly two hands



        

        results = {
            'join_params_1': config.join_len,
            #'join_params_2': .5,
            'discard_params_1': config.discard_len,
            #'discard_params_2': .5,
            'left_hand_segments': lists1Left,
            'right_hand_segments': lists1Right,
            'two_right_hand_segments': lists2Right,
            'two_left_hand_segments': lists2Left
        }
        print('these are the results to check (commented out rn)')
        #print(results)







        


        
        # # Addiiton to the final results
        # results['left_hand_segments'].append({'start_frame': start_frame, 
        #                                                 'end_frame': end_frame,
        #                                                 #'avg_confidence': 
                                                          
        #                                                   })
        

        # # Write sucess flag
        # logger.info("[Success] Processed hand landmarks {}".format(output_path))
        # with open(os.path.join(os.path.dirname(output_path), '.', f'.{video_name}_tracking_hand_landmarks_flag.txt'), 'w') as f:
        #     f.write('success')
        # continue


        # #Failure flag
        # logger.info("[Failure] Processed hand landmarks {}".format(output_path))
        # with open(os.path.join(os.path.dirname(output_path), '.', f'.{video_name}_tracking_hand_landmarks_flag.txt'), 'w') as f:
        #     f.write('failure')
        # continue
    
    
        # # End of the processing 
        print('Processed landmarks for {}'.format(output_path))

        
        # Save results and metrics
        with open(output_path, 'w') as f:
            json.dump(results, f, default=lambda x: x if isinstance(x, list) else int(x) if isinstance(x, np.integer) else x.__dict__)

        print('the results are saved i guess')    
        
        metrics.append({'join_len': config.join_len, 'discard_len': config.discard_len, 'input_path': input_path,  'tracking_hand_landmarks_compute_time': time.time() - start_time})
        if os.path.isfile(metrics_path):
            metricsdf = pd.concat([pd.read_csv(metrics_path), pd.DataFrame(metrics)])
        else:
            metricsdf =  pd.DataFrame(metrics)
        metricsdf.to_csv(metrics_path, index=False)
    
            
# Part I: Grouping the hands/ Tracking the hands

#this is the function, that is supposed to save the final version of the data. 
def plot_hand_detection_with_join_discard(copied_dataHandLandmark, join_len, discard_len, new_folder_path):
    #Hier I initialize lists to store frame indices, colors for plotting, and y-values for grouping frames
    frame_indices = []
    colors = []
    y_values = [] #

    #Now we iterate over each frame in the copied data
    for frame_index, frame in enumerate(copied_dataHandLandmark, start=1):
        #Dict. in order to count left and right hands
        hand_counts = {'left': 0, 'right': 0}
        valid_frame = True

        #check if 'handedness' data exists in the frame --> if not its an empty frame, then we will treat it as a valid one first and later check, if its score is sufficient.
        if 'handedness' in frame:
            for hand in frame['handedness']:
                try:
                    #Here: count left and right hands based on the handedness from the 'display_name'
                    hand_type = hand.get('display_name', 'unknown').lower()
                    if 'left' in hand_type:
                        hand_counts['left'] += 1
                    elif 'right' in hand_type:
                        hand_counts['right'] += 1
                    #Mark the frame as invalid if the hand score is less than 0.9 (as for these frames we have too little confidence that the data is "trustable")
                    if hand.get('score', 0) < 0.9:
                        valid_frame = False
                except (IndexError, KeyError) as e:
                    continue

            #to make the plot: append different colors depending on if they show over 1 left, over 1 right frame or something else. If you change the colors here, please make sure, to change them below as well.
            if not valid_frame:
                frame_indices.append(frame_index)
                colors.append('darkblue')
            elif hand_counts['left'] > 1:
                frame_indices.append(frame_index)
                colors.append('yellow')
            elif hand_counts['right'] > 1:
                frame_indices.append(frame_index)
                colors.append('red')
            else:
                frame_indices.append(frame_index)
                colors.append('darkblue')
        else:
            frame_indices.append(frame_index)
            colors.append('darkblue')
        #now we do this to make the plot more structured. We group it into packages of 1000 frames each for each row. ANd append the rows vertically. (starting with the 0 frame at the bottom of the plot)
        y_values.append((frame_index - 1) // 1000 + 1)

    #Now we create a list for the colors in the plot. So if you change the colors above, please also change here
    right_hand_indices = [i for i, color in enumerate(colors) if color == 'red']
    left_hand_indices = [i for i, color in enumerate(colors) if color == 'yellow']

    #printing statments
    print(f"Right hand indices before discarding: {right_hand_indices}")
    print(f"Left hand indices before discarding: {left_hand_indices}")

    # Discard small frame islands first
    discarded_right_hand_indices = join_and_discard(right_hand_indices, join_len=0, discard_len=discard_len)
    discarded_left_hand_indices = join_and_discard(left_hand_indices, join_len=0, discard_len=discard_len)

    # Plot after discard only
    plot_indices(frame_indices, y_values, colors, discarded_right_hand_indices, discarded_left_hand_indices, new_folder_path, "After Discard")

    # Printing statements after discard
    print(f"Right hand indices after discarding: {discarded_right_hand_indices}")
    print(f"Left hand indices after discarding: {discarded_left_hand_indices}")

    # Then, apply join to merge frames within the join length
    merged_right_hand_indices = join_and_discard(discarded_right_hand_indices, join_len=join_len, discard_len=0)
    merged_left_hand_indices = join_and_discard(discarded_left_hand_indices, join_len=join_len, discard_len=0)

    # Printing statements after join
    print(f"Right hand indices after joining: {merged_right_hand_indices}")
    print(f"Left hand indices after joining: {merged_left_hand_indices}")

    # Plot after join and discard
    plot_indices(frame_indices, y_values, colors, merged_right_hand_indices, merged_left_hand_indices, new_folder_path, "After Join and Discard")

    #deep copy of the original data to modify without altering the original data. To-do: remove/ clean some of these things, not needed here.
    modified_data = copy.deepcopy(copied_dataHandLandmark)

    #now we want to identify all the frames, that were either discarded or joined..
    joined_indices = merged_right_hand_indices + merged_left_hand_indices
    discarded_indices = [i for i in right_hand_indices + left_hand_indices if i not in joined_indices]

    #define a function to switch the handedness of a specific hand in a frame. (it is necassery to change both the display and the category name).
    def switch_handedness(frame_index, hand_index, new_label, reason):
        hands = modified_data[frame_index]['handedness']
        old_label = hands[hand_index]['display_name']
        hands[hand_index]['display_name'] = new_label
        hands[hand_index]['category_name'] = new_label
        print(f"{reason}: Switching handedness of frame {frame_index} hand {hand_index} from {old_label} to {new_label}")

    #Process frames for join
    for frame_index in merged_right_hand_indices:
        if 'handedness' in modified_data[frame_index]:
            hands = modified_data[frame_index]['handedness']
            if len(hands) == 2:
                if hands[0]['display_name'] != hands[1]['display_name']:
                    if hands[0]['score'] >= hands[1]['score']:
                        switch_handedness(frame_index, 1, hands[0]['display_name'], "Joining")
                    else:
                        switch_handedness(frame_index, 0, hands[1]['display_name'], "Joining")

    for frame_index in merged_left_hand_indices:
        if 'handedness' in modified_data[frame_index]:
            hands = modified_data[frame_index]['handedness']
            if len(hands) == 2:
                if hands[0]['display_name'] != hands[1]['display_name']:
                    if hands[0]['score'] >= hands[1]['score']:
                        switch_handedness(frame_index, 1, hands[0]['display_name'], "Joining")
                    else:
                        switch_handedness(frame_index, 0, hands[1]['display_name'], "Joining")

    #Now, process frames for discard
    for frame_index in discarded_indices:
        if 'handedness' in modified_data[frame_index]:
            hands = modified_data[frame_index]['handedness']
            if len(hands) == 2 and hands[0]['display_name'] == hands[1]['display_name']:
                if hands[0]['score'] > hands[1]['score']:
                    new_label = 'Left' if hands[0]['display_name'] == 'Right' else 'Right'
                    switch_handedness(frame_index, 1, new_label, "Discarding")
                else:
                    new_label = 'Left' if hands[1]['display_name'] == 'Right' else 'Right'
                    switch_handedness(frame_index, 0, new_label, "Discarding")

    frame_indices_merged = []
    colors_merged = []

    for frame_index in range(len(modified_data)):
        if frame_index in merged_right_hand_indices:
            frame_indices_merged.append(frame_index % 1000)
            colors_merged.append('red')
        elif frame_index in merged_left_hand_indices:
            frame_indices_merged.append(frame_index % 1000)
            colors_merged.append('yellow')
        else:
            frame_indices_merged.append(frame_index % 1000)
            colors_merged.append('darkblue')

    #Some printing, important atm to check that everything goes as planned, can later be removed to save time
    print("Sample frames from the initial data:")
    for i, frame in enumerate(copied_dataHandLandmark[0:10]):
        print(f"Frame {i+1}: {frame}")

    print("Sample frames from the modified data:")
    for i, frame in enumerate(modified_data[0:10]):
        print(f"Frame {i+1}: {frame}")

    print("Frames merged (joined):")
    for frame_index in merged_right_hand_indices:
        hands = copied_dataHandLandmark[frame_index]['handedness']
        modified_hands = modified_data[frame_index]['handedness']
        if hands != modified_hands:
            print(f"Right hand - Frame {frame_index} (Handedness adapted)")
        else:
            print(f"Right hand - Frame {frame_index} (No change)")

    for frame_index in merged_left_hand_indices:
        hands = copied_dataHandLandmark[frame_index]['handedness']
        modified_hands = modified_data[frame_index]['handedness']
        if hands != modified_hands:
            print(f"Left hand - Frame {frame_index} (Handedness adapted)")
        else:
            print(f"Left hand - Frame {frame_index} (No change)")

    print("Frames discarded:")
    for frame_index in discarded_indices:
        hands = copied_dataHandLandmark[frame_index]['handedness']
        modified_hands = modified_data[frame_index]['handedness']
        if hands != modified_hands:
            if frame_index in right_hand_indices:
                print(f"Right hand - Frame {frame_index} (Handedness adapted)")
            elif frame_index in left_hand_indices:
                print(f"Left hand - Frame {frame_index} (Handedness adapted)")
        else:
            if frame_index in right_hand_indices:
                print(f"Right hand - Frame {frame_index} (No change)")
            elif frame_index in left_hand_indices:
                print(f"Left hand - Frame {frame_index} (No change)")

    #plot..
    plt.figure(figsize=(15, 5))
    plt.scatter(frame_indices_merged, y_values, c=colors_merged, marker='|', label='Frames')
    plt.yticks(range(1, (max(y_values) + 1)))
    plt.xlabel('Frame Index')
    plt.ylabel('1000 Frame Groups')
    plt.title('Plot using modified data with join and discard function')
    extra_name = "PlotModifiedWithJoinAndDiscard"

    #legend
    legend_handles = [
        plt.Line2D([0], [0], color='darkblue', lw=4, label='No Hand Detected or Single Hand'),
        plt.Line2D([0], [0], color='yellow', lw=4, label='Multiple Left Hands Detected'),
        plt.Line2D([0], [0], color='red', lw=4, label='Multiple Right Hands Detected')
    ]
    plt.legend(handles=legend_handles, loc='upper right')

    fig_filename = os.path.join(new_folder_path, f"plot_{extra_name}.png")

    plt.savefig(fig_filename)
    plt.show()

    #Save modified data to a file
    modified_data_filename = os.path.join(new_folder_path, 'modified_dataJoinAndDiscard.json')
    with open(modified_data_filename, 'w') as f:
        json.dump(modified_data, f, indent=4)
    print(f"Modified data saved to {modified_data_filename}")
    
    if root_dir is None:
        root_dir = get_data_root()
        
                
    # -------- 
    # Define model and video dataset path
    model_name = 'tracking_hand_landmarks_v1'
    metrics_path = os.path.join(root_dir, 'dataframes', f'{get_host_name()}_compute_time_tracking_hand_landmarks.csv')
    
    # import operation config
    config = BaseSmartflatConfig()

    # Get video to process
    dset = get_dataset(dataset_name='hand_landmarks', root_dir=root_dir, scenario='gold_unprocessed_tracking')
    input_paths = dset.metadata.sort_values('size', ascending=True)['hand_landmarks_path'].tolist()
    
    print('Processing:')
    print('\n'.join(input_paths))
    
    metrics = [] 
    #for j, input_path in enumerate(input_paths[:10]): #TODO remove :1 
    for j, row in dset.metadata.dropna(subset=['fps']).iterrows():

        input_path = row.hand_landmarks_path
        fps = row.fps
         
        start_time = time.time()
        logger.info("Processing hand landmarks {}/{} for {}".format(j+1, len(input_paths), input_path))
        
        
        # 1) Initialization of the loop 
        output_path = fetch_output_path(input_path, model_name)
        video_name, _, _, _ = parse_path(input_path)

        print(f'Saving in {output_path}')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print('hello we are here')
        
        
        if os.path.isfile(output_path) and not overwrite:
            logger.info(f"Computation aready {output_path} (overwrite={overwrite})")
            continue
        print('hello we are here 2')
        try:
            print(f"Attempting to open file: {input_path}")
            with open(input_path, 'r') as f:
                raw_data = f.read()
                print("File read successfully, attempting to parse JSON")
                data = json.loads(raw_data)
                print(f"file was opened and parsed successfully: {input_path}")
        except MemoryError:
            logger.error(f"MemoryError: File {input_path} is too large, skipping.")
            continue
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in file {input_path}: {str(e)}, skipping this file...")
            continue
        except Exception as e:
            logger.error(f"Unexpected error when opening file {input_path}: {str(e)}, skipping this file...")
            continue

        print('hello we are here 3')

        deltaTimeJoin = 0.5 #seconds
        deltaTimeDiscard = 0.5 #seconds

        rowNumber = dset.metadata.loc[dset.metadata['hand_landmarks_path'] == input_path]
        #print(rowNumber)
        print('hello we are here 4')
        print(rowNumber)

        print(rowNumber['fps'])

        if not rowNumber.empty:
            fps = rowNumber['fps'].values[0]  # Get the fps value from the first matching row
            print('fps works')

        else:
            #print(f"No matching fps found for {input_paths}")
            continue
        print('hello we are here 5')
        
        join_len = deltaTimeJoin*fps
        discard_len = deltaTimeDiscard*fps


            


        
        # Strat of the processing


        print('the step 1 begins')
        # Step 1: Initial processing using DetectionAlgFin
        modified_data = DetectionAlgFin(data)

 

        # Step 2: Apply join and discard operation
        final_data, one_right_hand_indices, one_left_hand_indices, right_hand_indices, left_hand_indices = join_and_discard_handedness(modified_data, join_len, discard_len)

        # Process the segments
        lists1Right = find_connected_subseriesStartAndEnd(one_right_hand_indices)
        lists1Left = find_connected_subseriesStartAndEnd(one_left_hand_indices)
        lists2Right = find_connected_subseriesStartAndEnd(right_hand_indices)
        lists2Left = find_connected_subseriesStartAndEnd(left_hand_indices)


        lists1Left = loopFindScoreOneLeft(lists1Left, modified_data)
        lists1Right = loopFindScoreOneRight(lists1Right, modified_data)
        lists2Left = loopFindScoreTwoLeft(lists2Left, modified_data)    # Updated function
        lists2Right = loopFindScoreTwoRight(lists2Right, modified_data)  # Updated function


        # Now 'lists2Left' and 'lists2Right' contain only segments with exactly two hands



        

        results = {
            'join_params_1': config.join_len,
            #'join_params_2': .5,
            'discard_params_1': config.discard_len,
            #'discard_params_2': .5,
            'left_hand_segments': lists1Left,
            'right_hand_segments': lists1Right,
            'two_right_hand_segments': lists2Right,
            'two_left_hand_segments': lists2Left
        }
        print('these are the results to check (commented out rn)')
        #print(results)







        


        
        # # Addiiton to the final results
        # results['left_hand_segments'].append({'start_frame': start_frame, 
        #                                                 'end_frame': end_frame,
        #                                                 #'avg_confidence': 
                                                          
        #                                                   })
        

        # # Write sucess flag
        # logger.info("[Success] Processed hand landmarks {}".format(output_path))
        # with open(os.path.join(os.path.dirname(output_path), '.', f'.{video_name}_tracking_hand_landmarks_flag.txt'), 'w') as f:
        #     f.write('success')
        # continue


        # #Failure flag
        # logger.info("[Failure] Processed hand landmarks {}".format(output_path))
        # with open(os.path.join(os.path.dirname(output_path), '.', f'.{video_name}_tracking_hand_landmarks_flag.txt'), 'w') as f:
        #     f.write('failure')
        # continue
    
    
        # # End of the processing 
        print('Processed landmarks for {}'.format(output_path))

        
        # Save results and metrics
        with open(output_path, 'w') as f:
            json.dump(results, f, default=lambda x: x if isinstance(x, list) else int(x) if isinstance(x, np.integer) else x.__dict__)

        print('the results are saved i guess')    
        
        metrics.append({'join_len': config.join_len, 'discard_len': config.discard_len, 'input_path': input_path,  'tracking_hand_landmarks_compute_time': time.time() - start_time})
        if os.path.isfile(metrics_path):
            metricsdf = pd.concat([pd.read_csv(metrics_path), pd.DataFrame(metrics)])
        else:
            metricsdf =  pd.DataFrame(metrics)
        metricsdf.to_csv(metrics_path, index=False)
    
            
# Part I: Grouping the hands/ Tracking the hands

# Function to calculate the geometric center of the hand landmarks
def compute_barycenter(hand_landmarks):
    x_coords = [landmark['x'] for landmark in hand_landmarks]
    y_coords = [landmark['y'] for landmark in hand_landmarks]
    z_coords = [landmark['z'] for landmark in hand_landmarks]
    return {'x': np.mean(x_coords), 'y': np.mean(y_coords), 'z': np.mean(z_coords)}

# Function to calculate the Euclidean distance between two barycenters
def calculate_distance(barycenter1, barycenter2):
    return np.sqrt((barycenter1['x'] - barycenter2['x']) ** 2 +
                   (barycenter1['y'] - barycenter2['y']) ** 2 +
                   (barycenter1['z'] - barycenter2['z']) ** 2)

# Function to track hands across frames and assign unique indices to each hand
def track_hands_with_handedness(data_list):
    hand_id_counter = 0
    previous_frame_hands = []
    tracked_frames = []

    for frame_index, frame in enumerate(data_list):
        current_frame_hands = []

        for hand_index, hand_data in enumerate(frame['hand_landmarks']):
            barycenter = compute_barycenter(hand_data)
            handedness = frame['handedness'][hand_index][0]['category_name']
            score = frame['handedness'][hand_index][0]['score']
            current_frame_hands.append({'barycenter': barycenter, 'index': None, 'handedness': handedness, 'score': score})

        if not previous_frame_hands:
            for hand in current_frame_hands:
                hand['index'] = hand_id_counter
                hand_id_counter += 1
        else:
            used_prev_indices = set()

            for current_hand in current_frame_hands:
                min_distance = float('inf')
                closest_hand_index = None

                for prev_hand in previous_frame_hands:
                    if prev_hand['index'] not in used_prev_indices:
                        distance = calculate_distance(current_hand['barycenter'], prev_hand['barycenter'])
                        if distance < min_distance:
                            min_distance = distance
                            closest_hand_index = prev_hand['index']

                if closest_hand_index is not None:
                    current_hand['index'] = closest_hand_index
                    used_prev_indices.add(closest_hand_index)
                else:
                    current_hand['index'] = hand_id_counter
                    hand_id_counter += 1

        tracked_frames.append(current_frame_hands)
        previous_frame_hands = current_frame_hands

    return tracked_frames

# Function to calculate the weighted average and dominant handedness
def calculate_weighted_average_and_majority_hand(df):
    grouped = df.groupby('index')
    weighted_scores = {}
    for index, group in grouped:
        right_handedness = group[group['handedness'] == 'Right']
        left_handedness = group[group['handedness'] == 'Left']
        right_count = len(right_handedness)
        left_count = len(left_handedness)

        right_percentage = right_count / len(group) * 100
        left_percentage = left_count / len(group) * 100

        if right_percentage >= left_percentage:
            dominant_handedness = 'Right'
            dominant_score = right_handedness['score'].sum() / right_count if right_count > 0 else 0
        else:
            dominant_handedness = 'Left'
            dominant_score = left_handedness['score'].sum() / left_count if left_count > 0 else 0

        weighted_scores[index] = {
            'handedness': dominant_handedness,
            'score': dominant_score
        }
    return weighted_scores

# Function to update the original data with the new handedness and scores
def update_data_with_weighted_scores(original_data, weighted_scores, tracked_hands):
    for frame_index, frame in enumerate(original_data):
        if frame_index < len(tracked_hands):
            for hand_index, hand_data in enumerate(frame['hand_landmarks']):
                tracked_hand = tracked_hands[frame_index][hand_index]
                hand_id = tracked_hand['index']

                frame['handedness'][hand_index][0]['category_name'] = weighted_scores[hand_id]['handedness']
                frame['handedness'][hand_index][0]['display_name'] = weighted_scores[hand_id]['handedness']
                frame['handedness'][hand_index][0]['score'] = weighted_scores[hand_id]['score']
    
    return original_data

# Function to create a DataFrame from tracked hands
def create_hand_dataframe(tracked_hands):
    hand_data = []
    for frame in tracked_hands:
        for hand in frame:
            hand_data.append({
                'index': hand['index'],
                'handedness': hand['handedness'],
                'score': hand['score'],
                'barycenter': hand['barycenter']
            })
    df = pd.DataFrame(hand_data)
    return df

# Main function to execute the algorithm
def DetectionAlgFin(dataHandLandmark):
    tracked_hands_with_handedness = track_hands_with_handedness(dataHandLandmark)
    df = create_hand_dataframe(tracked_hands_with_handedness)
    weighted_scores = calculate_weighted_average_and_majority_hand(df)
    modified_dataHandLandmark = update_data_with_weighted_scores(dataHandLandmark, weighted_scores, tracked_hands_with_handedness)
    return modified_dataHandLandmark

# Part III: Join and Discard

def join_and_discard(frame, join_len, discard_len, binary_mask=False):
    if binary_mask:
        original_size = len(frame)
        frame = np.argwhere(frame == 1).squeeze()

    try:
        if len(frame) == 0:
            return np.zeros(original_size) if binary_mask else []
    except:
        return np.zeros(original_size) if binary_mask else []

    # First join
    frame = sorted(frame)
    joined_frame = []
    prev = frame[0]
    for f in frame[1:]:
        if f > prev + 1 and f <= prev + join_len:
            joined_frame.extend(list(range(prev + 1, f)))
        prev = f

    frame = sorted(frame + joined_frame)

    # Then discard
    discard_frame = []
    prev = frame[0]
    island = [prev]
    for f in frame[1:]:
        if f == prev + 1:
            island.append(f)
        else:
            if len(island) <= discard_len:
                discard_frame.extend(island)
            island = [f]
        prev = f

    if len(island) <= discard_len:
        discard_frame.extend(island)

    new_frame = [f for f in frame if f not in discard_frame]

    if binary_mask:
        new_mask = np.zeros(original_size)
        new_mask[new_frame] = 1
        return new_mask
    
    return new_frame

def join_and_discard_handedness(data, join_len, discard_len):

    

    right_hand_indices = []
    left_hand_indices = []
    one_right_hand_indices = []
    one_left_hand_indices = []
    

    # Identify right and left hand indices
    for frame_index, frame in enumerate(data):
        hand_counts = {'left': 0, 'right': 0}

        if 'handedness' in frame:
            #print("frame['handedness']:", frame['handedness'])
            for hand_list in frame['handedness']:
                hand = hand_list[0] 
                #print("hand:", hand)
                hand_type = hand.get('display_name', 'unknown').lower()
                if 'left' in hand_type:
                    hand_counts['left'] += 1
                elif 'right' in hand_type:
                    hand_counts['right'] += 1

            if hand_counts['left'] == 1:
                one_left_hand_indices.append(frame_index)
            if hand_counts['right'] == 1:
                one_right_hand_indices.append(frame_index)
            if hand_counts['left'] > 1:
                left_hand_indices.append(frame_index)
            elif hand_counts['right'] > 1:
                right_hand_indices.append(frame_index)



    # Discard small frame islands
    discarded_right_hand_indices = join_and_discard(right_hand_indices, join_len=0, discard_len=discard_len)
    discarded_left_hand_indices = join_and_discard(left_hand_indices, join_len=0, discard_len=discard_len)

    # Join frames within the join length
    merged_right_hand_indices = join_and_discard(discarded_right_hand_indices, join_len=join_len, discard_len=0)
    merged_left_hand_indices = join_and_discard(discarded_left_hand_indices, join_len=join_len, discard_len=0)

    # Modify data based on the merged and discarded indices
    def switch_handedness(frame_index, hand_index, new_label):
        hands = data[frame_index]['handedness']
        old_label = hands[hand_index][0]['display_name']
        
        # Only switch if the old label is different from the new label
        if old_label != new_label:
            hands[hand_index][0]['display_name'] = new_label
            hands[hand_index][0]['category_name'] = new_label
            print(f"Switched handedness in frame {frame_index}, hand {hand_index} from {old_label} to {new_label}")

        

    for frame_index in merged_right_hand_indices:
        if 'handedness' in data[frame_index]:
            hands = data[frame_index]['handedness']
            if len(hands) == 2 and hands[0][0]['display_name'] != hands[1][0]['display_name']:
                if hands[0][0]['score'] >= hands[1][0]['score']:
                    switch_handedness(frame_index, 1, hands[0][0]['display_name'])
                else:
                    switch_handedness(frame_index, 0, hands[1][0]['display_name'])

    for frame_index in merged_left_hand_indices:
        if 'handedness' in data[frame_index]:
            hands = data[frame_index]['handedness']
            if len(hands) == 2 and hands[0][0]['display_name'] != hands[1][0]['display_name']:
                if hands[0][0]['score'] >= hands[1][0]['score']:
                    switch_handedness(frame_index, 1, hands[0][0]['display_name'])
                else:
                    switch_handedness(frame_index, 0, hands[1][0]['display_name'])

    # Process frames for discard
    discarded_indices = [i for i in right_hand_indices + left_hand_indices if i not in merged_right_hand_indices + merged_left_hand_indices]

    for frame_index in discarded_indices:
        if 'handedness' in data[frame_index]:
            hands = data[frame_index]['handedness']
            if len(hands) == 2 and hands[0][0]['display_name'] == hands[1][0]['display_name']:
                if hands[0][0]['score'] > hands[1][0]['score']:
                    new_label = 'Left' if hands[0][0]['display_name'] == 'Right' else 'Right'
                    switch_handedness(frame_index, 1, new_label)
                else:
                    new_label = 'Left' if hands[1][0]['display_name'] == 'Right' else 'Right'
                    switch_handedness(frame_index, 0, new_label)

    return data, one_right_hand_indices, one_left_hand_indices, right_hand_indices, left_hand_indices

def find_connected_subseriesStartAndEnd(indices):
    if not indices:
        return []

    # Convert the list to a numpy array
    indices = np.array(indices)
    
    # Find the breaks where the difference between consecutive indices is greater than 1
    breaks = np.where(np.diff(indices) > 1)[0]
    
    # Split the array at the breaks
    subseries = np.split(indices, breaks + 1)
    
    # Convert the resulting subarrays back to lists
    subseries = [list(sub) for sub in subseries]

    startAndEndSeries = [[sub[0],sub[-1]] for sub in subseries]
    
    return startAndEndSeries

def find_right_hand_score(frame):
    if 'handedness' in frame:
        for hand_list in frame['handedness']:
            hand = hand_list[0]  # Extract the dictionary from the list
            if hand.get('category_name', '').lower() == 'right':
                return hand.get('score', None)
    return None


def find_left_hand_score(frame):
    if 'handedness' in frame:
        for hand_list in frame['handedness']:
            hand = hand_list[0]  # Extract the dictionary from the list
            if hand.get('category_name', '').lower() == 'left':
                return hand.get('score', None)
    return None

def find_two_right_hand_scores(frame):
    right_hand_scores = []
    if 'handedness' in frame:
        for hand_list in frame['handedness']:
            hand = hand_list[0]  # Extract the dictionary from the list
            if hand.get('category_name', '').lower() == 'right':
                right_hand_scores.append(hand.get('score', None))
    # Return scores only if exactly two right hands are detected
    if len(right_hand_scores) == 2:
        return right_hand_scores
    return None



def find_two_left_hand_scores(frame):
    left_hand_scores = []
    if 'handedness' in frame:
        for hand_list in frame['handedness']:
            hand = hand_list[0]  # Extract the dictionary from the list
            if hand.get('category_name', '').lower() == 'left':
                left_hand_scores.append(hand.get('score', None))
    # Return scores only if exactly two left hands are detected
    if len(left_hand_scores) == 2:
        return left_hand_scores
    return None



def loopFindScoreOneLeft(lists, data):
    for element in lists:
        frame = element[0]
        # Debugging: Print the type and content of `frame`
        print(f"Type of frame: {type(frame)}, frame content: {frame}")

        if isinstance(frame, (int, np.int64)):
            frame = data[frame]  # Get the actual frame data using the index

        # Continue with the function
        score = find_left_hand_score(frame)
        element.insert(2, score)
    return lists


        
def loopFindScoreOneRight(lists, data):
    for element in lists:
        frame = element[0]
        # Print to debug if necessary
        print(f"Type of frame: {type(frame)}, frame content: {frame}")

        if isinstance(frame, (int, np.int64)):
            frame = data[frame]  # Retrieve the actual frame data using the index

        score = find_right_hand_score(frame)
        element.insert(2, score)
    return lists


def loopFindScoreTwoLeft(lists, data):
    # Create a new list to hold only the valid segments
    valid_segments = []
    for element in lists:
        frame_index = element[0]
        if isinstance(frame_index, (int, np.int64)):
            frame = data[frame_index]
        else:
            continue  # Skip if frame index is invalid
        scores = find_two_left_hand_scores(frame)
        if scores is not None:
            # Only include segments where exactly two left hands are detected
            element_with_scores = element.copy()
            element_with_scores.insert(2, scores)
            valid_segments.append(element_with_scores)
    return valid_segments




def loopFindScoreTwoRight(lists, data):
    # Create a new list to hold only the valid segments
    valid_segments = []
    for element in lists:
        frame_index = element[0]
        if isinstance(frame_index, (int, np.int64)):
            frame = data[frame_index]
        else:
            continue  # Skip if frame index is invalid
        scores = find_two_right_hand_scores(frame)
        if scores is not None:
            # Only include segments where exactly two right hands are detected
            element_with_scores = element.copy()
            element_with_scores.insert(2, scores)
            valid_segments.append(element_with_scores)
    return valid_segments



def openjson(path):
    try:
        print(f"Opening JSON file: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"File loaded successfully: {path}")
        return data
    except MemoryError as me:
        print(f"MemoryError while opening JSON file: {path}")
        raise me
    except json.JSONDecodeError as je:
        print(f"JSONDecodeError in file: {path} - {str(je)}")
        raise je
    except Exception as e:
        print(f"Exception in openjson: {str(e)}")
        raise e  # Re-raise to be caught at a higher level if needed



def openjson(path):
    try:
        print(f"Opening JSON file: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"File loaded successfully: {path}")
        return data
    except MemoryError as me:
        print(f"MemoryError while opening JSON file: {path}")
        raise me
    except json.JSONDecodeError as je:
        print(f"JSONDecodeError in file: {path} - {str(je)}")
        raise je
    except Exception as e:
        print(f"Exception in openjson: {str(e)}")
        raise e  # Re-raise to be caught at a higher level if needed


#old code end.

def get_args():
    parser = argparse.ArgumentParser(
        'Perform hands landmarks processing', add_help=False)
    parser.add_argument('-r', '--root_dir', default='local', help='local or harddrive')
    parser.add_argument('-o', '--overwrite', action="store_true", default=False, help='Orewrite output if existing')
    #TODOALEX: do we need more ? 
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()    
    
    if args.root_dir == 'local':
        
        root_dir = None
        
    elif args.root_dir == 'harddrive':
        
        root_dir = '/Volumes/Smartflat/data' 
        
    print('root_dir:', root_dir)
    main(root_dir=root_dir, overwrite=args.overwrite)

    print('Done')
    sys.exit(0)
