"""Smartflat gaze datasets classes and parsing function."""



import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from IPython.display import display


from smartflat.constants import (
    gaze_columns_mapping,
    gaze_features,
    gaze_type_columns_mapping,
    gaze_useful_columns,
)
from smartflat.utils.utils import pairwise
from smartflat.utils.utils_coding import green, red
from smartflat.utils.utils_io import get_data_root

# def parse_tobii_data(filename):        
#     filename = filename
#     dir_name = os.path.dirname(filename)
    
#     raw_data = pd.read_csv(filename, sep='\t')# usecols = gaze_useful_columns)
#     raw_data.rename(columns=gaze_columns_mapping, errors="raise", inplace=True)
    
#     if raw_data['gaze_direction_R_x'].dtypes == "O":
#         raw_data = preprocess_data(raw_data)       
        
#     raw_data.loc[:,'time'] = raw_data['time']*0.000001 # set back to seconds
#     raw_data.loc[:,'event_duration'] = raw_data['event_duration'] / 1000 # set back to seconds

#     data_gaze = raw_data[raw_data['sensor']=='Eye Tracker'][['time', 'gaze_2D_x', 'gaze_2D_y', 'gaze_3D_x', 'gaze_3D_y',
#                                                         'gaze_3D_z', 'gaze_direction_L_x', 'gaze_direction_L_y',
#                                                         'gaze_direction_L_z', 'gaze_direction_R_x', 'gaze_direction_R_y',
#                                                         'gaze_direction_R_z', 'pupil_L_x', 'pupil_L_y', 'pupil_L_z',
#                                                         'pupil_R_x', 'pupil_R_y', 'pupil_R_z', 'pupil_L_diameter',
#                                                         'pupil_R_diameter', 'valid_L', 'valid_R', 
#                                                         #'eye_movement_type', 'event_duration', 'eye_movement_type_index', 
#                                                         'fixation_point_x','fixation_point_y']]

#     data_gaze_event = raw_data[raw_data['sensor'].isna()][['time', 'eye_movement_type','event_duration',
#                                                     'eye_movement_type_index', 'fixation_point_x',
#                                                     'fixation_point_y']]



#     data_accelerometer = raw_data[raw_data['sensor']=='Accelerometer'][['time', 'eye_movement_type', 'event_duration', 
#                                                                 'eye_movement_type_index', 'fixation_point_x',
#                                                                 'fixation_point_y', 'acceleration_x',
#                                                                 'acceleration_y', 'acceleration_z']]


#     data_gyroscope = raw_data[raw_data['sensor']=='Gyroscope'][['time', 'eye_movement_type','event_duration', 
#                                                         'eye_movement_type_index', 'fixation_point_x',
#                                                         'fixation_point_y', 'gyro_x', 'gyro_y', 'gyro_z']]
#     data_gaze.reset_index(drop=True, inplace=True)
#     data_gaze_event.reset_index(drop=True, inplace=True)
#     data_accelerometer.reset_index(drop=True, inplace=True)
#     data_gyroscope.reset_index(drop=True, inplace=True)
    
#     #data_gaze.loc[:, 'gaze_direction_R_x'] = gazeDataset.data_gaze['gaze_direction_R_x'].apply(lambda x: float(x.replace(',', '.')) if not pd.isnull(x) else x)

#     return data_gaze, data_gaze_event, data_accelerometer, data_gyroscope

def parse_tobii_data(filename, data_type='all'):
    raw_data = pd.read_csv(filename, sep='\t', low_memory=False) 
    raw_data.rename(columns=gaze_columns_mapping, errors="raise", inplace=True)

    raw_data = preprocess_data(raw_data)

    raw_data['time'] = raw_data['time'] * 0.000001  # seconds
    raw_data['event_duration'] = raw_data['event_duration'] / 1000  # seconds
    
    raw_data['duration_gaze'] = raw_data.gaze_duration.apply(lambda x: x / 1000 /60)
    
    raw_data['valid_R'].replace({'Invalid': 0, 'Valid': 1}, inplace=True)
    raw_data['valid_L'].replace({'Invalid': 0, 'Valid': 1}, inplace=True)

    #gdata.dropna(subset=['eye_movement_type'], inplace=True)
    na_count = raw_data['eye_movement_type'].isna().sum()
    print(f'Filled {na_count} missing values in eye_movement_type')
    raw_data['eye_movement_type'].fillna('Unclassified', inplace=True)
    #print(f'Dropped {n - len(gaze_event_data)} rows with missing eye_movement_type')

    # Define datasets based on data_type
    dataframes = {
        'gaze_data': raw_data[raw_data['sensor'] == 'Eye Tracker'][[
            'time', 'duration_gaze', 'gaze_2D_x', 'gaze_2D_y', 'gaze_3D_x', 'gaze_3D_y',
            'gaze_3D_z', 'gaze_direction_L_x', 'gaze_direction_L_y', 'gaze_direction_L_z',
            'gaze_direction_R_x', 'gaze_direction_R_y', 'gaze_direction_R_z',
            'pupil_L_x', 'pupil_L_y', 'pupil_L_z', 'pupil_R_x', 'pupil_R_y',
            'pupil_R_z', 'pupil_L_diameter', 'pupil_R_diameter', 'valid_L',
            'valid_R', 'fixation_point_x', 'fixation_point_y'
        ]],
        'gaze_event_data': raw_data[raw_data['sensor'].isna()][[
            'time', 'eye_movement_type', 'event_duration', 'eye_movement_type_index',
            'fixation_point_x', 'fixation_point_y'
        ]],
        'accelerometric_data': raw_data[raw_data['sensor'] == 'Accelerometer'][[
            'time', 'acceleration_x', 'acceleration_y',
            'acceleration_z'
        ]],
        'gyroscopic_data': raw_data[raw_data['sensor'] == 'Gyroscope'][[
            'time','gyro_x', 'gyro_y', 'gyro_z'
        ]]
    }
    if len(raw_data['time']) == 0:
        print(f'/!\ No gaze data after parsing file: {filename}')
        return []
    
    # dataframes['gaze_data']['gaze_2D_x'] = dataframes['gaze_data']['gaze_2D_x'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)
    # dataframes['gaze_data']['gaze_2D_y'] = dataframes['gaze_data']['gaze_2D_y'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)
    # dataframes['gaze_data']['gaze_3D_x'] = dataframes['gaze_data']['gaze_3D_x'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)
    # dataframes['gaze_data']['gaze_3D_y'] = dataframes['gaze_data']['gaze_3D_y'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)
    # dataframes['gaze_data']['gaze_3D_z'] = dataframes['gaze_data']['gaze_3D_z'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)
    # dataframes['gaze_data']['pupil_L_diameter'] = dataframes['gaze_data']['pupil_L_diameter'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)
    # dataframes['gaze_data']['pupil_R_diameter'] = dataframes['gaze_data']['pupil_R_diameter'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)
    # dataframes['gaze_data']['fixation_point_x'] = dataframes['gaze_data']['fixation_point_x'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)
    # dataframes['gaze_data']['fixation_point_y'] = dataframes['gaze_data']['fixation_point_y'].apply(lambda x: x.replace(',', '.') if pd.notna(x) else np.nan).astype(float)

    if data_type == 'all':
        return {key: df.reset_index(drop=True) for key, df in dataframes.items()}
    if data_type in dataframes:
        return dataframes[data_type].reset_index(drop=True)
    raise ValueError(f"Invalid data_type '{data_type}'. Choose from {list(dataframes.keys()) + ['all']}.")

def preprocess_data(raw_data):
    for col in raw_data.columns:
        try:
            raw_data.loc[:, col] = raw_data[col].apply(lambda x: float(x.replace(',', '.')) if pd.notna(x) else x)
        except :
            pass
        
    return raw_data
        
      
def compute_segments_gaze_features(df, cpts_col='cpts', verbose=False):
    
    func_dict = {'gaze_data': compute_segments_gaze_features_row,
                 'gaze_event_data': compute_segments_gaze_event_features_row,
                 'accelerometric_data': compute_segments_accelerometric_features_row,
                 'gyroscopic_data': compute_segments_gyroscopic_features_row}
    
    
    #df['cpts_frames'] = df.apply(lambda row: [int(cpt * row.n_frames / row.N) for cpt in row.cpts], axis=1)
    df['cpts_frames'] = df.apply(lambda row: [int((row.test_bounds[0] + cpt) * row.n_frames / row.N_raw) for cpt in row[cpts_col]], axis=1)
    df['cpts_temporal'] = df.apply(lambda row: [cpt / row.fps / 60 for cpt in row.cpts_frames], axis=1)
    df['segments_bounds'] = df.cpts_temporal.apply(lambda x: [(s, e) for s, e in pairwise(x)])


    for data_type in ['gaze_event_data', 'gaze_data',  'accelerometric_data', 'gyroscopic_data']:
        print(f'Populating raw features for {data_type}...')
        df = populate_gaze_data(df, data_type=data_type, verbose=verbose)
        
        # if (data_type != 'gaze_event_data') and not (df['time'].apply(lambda x: 0 if not isinstance(x, list) else np.abs(x[-2] - x[-1])).apply(lambda x: x < 0.1).mean() == 1):
        #     print('Warning: Time is not consistent')
        #     display(df[df['time'].apply(lambda x: 0 if not isinstance(x, list) else  np.abs(x[-2] - x[-1])).apply(lambda x: x > 0.1)])
         
        print(f'Computing {data_type} features ...')
        df[gaze_features[data_type]] = df.apply(func_dict[data_type], axis=1)
        df['has_gaze'] = df.time.apply(lambda x: int(isinstance(x, list)))
        df.drop(columns=gaze_type_columns_mapping[data_type], inplace=True)
        print(f'Populated {data_type} features')
        print('columns:', df.columns)
        green(f'Done!')
        
    # df.drop(columns=['cpts_frames', 'cpts_temporal'], inplace=True)
    # df[gaze_features['gaze_data']] = df.apply(compute_segments_gaze_features_row, axis=1)
    # df[gaze_features['gaze_event_data']] = df.apply(compute_segments_gaze_event_features_row, axis=1)
    # df[gaze_features['accelerometric_data']] = df.apply(compute_segments_accelerometric_features_row, axis=1)
    # df[gaze_features['gyroscopic_data']] = df.apply(compute_segments_gyroscopic_features_row, axis=1)

    return df
  
def populate_gaze_data(df, data_type='gaze_event_data', verbose=False):
    
    def populate_gaze_data_row(row, data_type='gaze_event_data'):
        
        if row.has_gaze == 0:
            #print('No gaze data for participant', row.participant_id)
            
            return pd.Series({f: np.nan for f in gaze_type_columns_mapping[data_type]})

        # Define the path to search for gaze data
        gaze_data_list = glob(
            os.path.join(
                os.path.dirname(get_data_root(local=True)),
                'data-gaze', '*', f'*{row.participant_id}*'
            )
        )
        # p = 0.1
        # if np.random.rand() >  p: # Minim no gaze data for debugging
        #     red(f"No gaze data found for participant {row.participant_id} - data_type=")
        #     #return pd.DataFrame(columns=gaze_type_columns_mapping[data_type], index=[0])
        #     return pd.Series({col: [] for col in gaze_type_columns_mapping[data_type]})
        
        # Handle cases of missing or multiple files
        if len(gaze_data_list) == 0:
            red(f"No gaze data found for participant {row.participant_id} - data_type=")
            #return pd.DataFrame(columns=gaze_type_columns_mapping[data_type], index=[0])
            return pd.Series({col: [] for col in gaze_type_columns_mapping[data_type]})
        elif len(gaze_data_list) == 1:
            gaze_path = gaze_data_list[0]
            green(f"Found gaze data: {gaze_path}")
        else:
            raise ValueError(f"Multiple gaze data files found for participant {row.participant_id}")

        # Parse the gaze data
        gdata = parse_tobii_data(gaze_path, data_type=data_type)
        if len(gdata) == 0:
            return pd.Series({f: np.nan for f in gaze_type_columns_mapping[data_type]})
        gdata = gdata.assign(participant_id=row.participant_id)

        # Print details
        print(f"Participant {row.participant_id}: {len(gdata)} rows, max time = {gdata['time'].max() / 60:.2f} min")
        
        # Group by participant ID, aggregate lists, and reset index
        return gdata.groupby('participant_id').agg(list).reset_index(drop=True).iloc[0]

    #df[['time', 'eye_movement_type', 'event_duration', 'eye_movement_type_index', 'fixation_point_x', 'fixation_point_y']] = df.apply(populate_gaze_data_row, axis=1, data_type='gaze_event_data')
    
    
    # Ensure populate_gaze_data_row returns a Series with appropriate column names
    df[gaze_type_columns_mapping[data_type]] = df.apply(populate_gaze_data_row, data_type=data_type, axis=1)
        
    
    df['duration_gaze'] = df.time.apply(lambda x: np.max(x) / 60 if isinstance(x, list) and x else np.nan)
    df['gaze_video_tobii_duration_diff'] = df.apply(lambda x: x.duration_gaze - x.duration if not np.isnan(x.duration_gaze) else  np.nan, axis=1)
    
    if verbose:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 3))
        df.drop_duplicates(subset=['participant_id']).gaze_video_tobii_duration_diff.hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of gaze - video Difference', fontsize=16); ax.set_xlabel('Duration Difference', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14); ax.grid(True, linestyle='--', alpha=0.7); 
        plt.savefig('gaze_video_duration_diff.png')
        plt.show()

    #n = df.participant_id.nunique()
    #df = df[ () (df['duration_diff'].abs() < 2)]
    #print(f'Discarded {n - df.participant_id.nunique()}/{n} participant data because of mismatch between gaze and video duration (tol=2 min)')

    green(f'Loaded gaze data for {df.participant_id.nunique()} participants and {df.duration_gaze.sum():.2f} minutes of gaze data ({df.task_name.nunique()} tasks.)')

    return df

# Functions to compute the gaze and occulemetric features using the segments bounds 


def compute_segments_gaze_event_features_row(row):

    
    if row.has_gaze == 0:
        return pd.Series({f:np.nan for f in gaze_features['gaze_event_data']})

    res = {f:[] for f in gaze_features['gaze_event_data']}
    
    fixation_duration_list_all = []
    saccade_duration_list_all = []
    fixation_x_list_all = []
    fixation_y_list_all = []
    
    
    # Segments bunds are in minutes and time in seconds 
    for idx_segment, (s, e) in enumerate(row.segments_bounds):
        
        #print(idx_segment, (s, e))
        n_saccades = 0
        n_fixation = 0
        saccade_duration_list = []
        fixation_duration_list = []
        fixation_x_list = []
        fixation_y_list = []

        for t, time in enumerate(row.time):
            
            if ((time / 60) >= s) and ((time / 60) <= e):
                
                eye_movement_type = row.eye_movement_type[t]
                event_duration = row.event_duration[t]
                event_duration = row.event_duration[t]
                fixation_point_x = row.fixation_point_x[t]
                fixation_point_y = row.fixation_point_y[t]
                
                if eye_movement_type == 'Saccade':
                    n_saccades+=1
                    saccade_duration_list.append(event_duration)
                elif eye_movement_type == 'Fixation':
                    n_fixation +=1
                    fixation_duration_list.append(event_duration)
                    fixation_x_list.append(fixation_point_x)
                    fixation_y_list.append(fixation_point_y)

                

                #print(time_min, eye_movement_type, event_duration, fixation_point_x, fixation_point_y)
                
        res['n_saccades'].append(n_saccades)
        res['n_fixation'].append(n_fixation)
        res['saccade_duration_mean'].append(np.nanmean(saccade_duration_list) if len(saccade_duration_list) > 0 else np.nan)
        res['saccade_duration_std'].append(np.nanstd(saccade_duration_list) if len(saccade_duration_list) > 0 else np.nan)
        res['fixation_duration_mean'].append(np.nanmean(fixation_duration_list) if len(fixation_duration_list) > 0 else np.nan)
        res['fixation_duration_std'].append(np.nanstd(fixation_duration_list) if len(fixation_duration_list) > 0 else np.nan)

        res['saccade_frequency'].append(n_saccades / (e - s) * 60 )
        res['fixation_frequency'].append(n_fixation / (e - s) * 60 )
        
        res['fixation_x_mean'].append(np.nanmean(fixation_x_list) if len(fixation_x_list) > 0 else np.nan)
        res['fixation_x_std'].append(np.nanstd(fixation_x_list) if len(fixation_x_list) > 0 else np.nan)
        res['fixation_y_mean'].append(np.nanmean(fixation_y_list) if len(fixation_y_list) > 0 else np.nan)
        res['fixation_y_std'].append(np.nanstd(fixation_y_list) if len(fixation_y_list) > 0 else np.nan)
        
        saccade_duration_list_all.extend(saccade_duration_list)
        fixation_duration_list_all.extend(fixation_duration_list)
        fixation_x_list_all.extend(fixation_x_list)
        fixation_y_list_all.extend(fixation_y_list)
        
    res['n_saccade_tot'] = np.sum(res['n_saccades'])
    res['n_fixation_tot'] = np.sum(res['n_fixation'])
    res['fixation_duration_tot_mean'] = np.nanmean(fixation_duration_list_all) 
    res['fixation_duration_tot_std'] = np.nanstd(fixation_duration_list_all)
    res['saccade_duration_tot_mean'] = np.nanmean(saccade_duration_list_all)
    res['saccade_duration_tot_std'] = np.nanstd(saccade_duration_list_all)
    
    if len(row.time) == 0:
        max_time = np.nan
    
    elif not ((np.abs(row.time[-2] - row.time[-1]))  < 0.1):
        print('Error in time')
        print(row.time[-5:])
        max_time = row.time[-2]
    else:
        max_time = row.time[-1]

    res['saccade_frequency_tot'] = res['n_saccade_tot'] /  max_time
    res['fixation_frequency_tot'] = res['n_fixation_tot'] /  max_time
    
    res['fixation_x_tot_mean'] = np.nanmean(fixation_x_list_all) 
    res['fixation_x_tot_std'] = np.nanstd(fixation_x_list_all) 

    res['fixation_y_tot_mean'] = np.nanmean(fixation_y_list_all) 
    res['fixation_y_tot_std'] = np.nanstd(fixation_y_list_all) 
    
    
    # For the records, previous way of computing the all-samples gaze features from the raw data

    # # Event-gaze-data features 
    # results['n_saccade_tot'] = results.apply(lambda x: int(np.sum([1 for t, eye_m  in enumerate(x.eye_movement_type) if (eye_m == 'Saccade')])), axis=1)
    # results['n_fixation_tot'] = results.apply(lambda x: int(np.sum([1 for t, eye_m  in enumerate(x.eye_movement_type) if (eye_m == 'Fixation')])), axis=1)

    # results['fixation_duration_tot_mean'] = results.apply(lambda x: np.nanmean([event_duration for t, event_duration  in enumerate(x.event_duration) if (x.eye_movement_type[t] == 'Fixation')]), axis=1)
    # results['fixation_duration_tot_std'] = results.apply(lambda x: np.nanstd([event_duration for t, event_duration  in enumerate(x.event_duration) if (x.eye_movement_type[t] == 'Fixation')]), axis=1)

    # results['saccade_duration_tot_mean'] = results.apply(lambda x: np.nanmean([event_duration for t, event_duration  in enumerate(x.event_duration) if (x.eye_movement_type[t] == 'Saccade')]), axis=1)
    # results['saccade_duration_tot_std'] = results.apply(lambda x: np.nanstd([event_duration for t, event_duration  in enumerate(x.event_duration) if (x.eye_movement_type[t] == 'Saccade')]), axis=1)

    # results['saccade_frequency_tot'] = results.apply(lambda x: x.n_saccade_tot / np.max(x.time), axis=1)
    # results['fixation_frequency_tot'] = results.apply(lambda x: x.n_fixation_tot / np.max(x.time), axis=1)

    # results['fixation_x_tot_mean'] = results.apply(lambda x: np.nanmean([fixation_x for t, fixation_x  in enumerate(x.fixation_point_x) if (x.eye_movement_type[t] == 'Fixation')]), axis=1)
    # results['fixation_x_tot_std'] = results.apply(lambda x: np.nanstd([fixation_x for t, fixation_x  in enumerate(x.fixation_point_x) if (x.eye_movement_type[t] == 'Fixation')]), axis=1)

    # results['fixation_y_tot_mean'] = results.apply(lambda x: np.nanmean([fixation_y for t, fixation_y  in enumerate(x.fixation_point_y) if (x.eye_movement_type[t] == 'Fixation')]), axis=1)
    # results['fixation_y_tot_std'] = results.apply(lambda x: np.nanstd([fixation_y for t, fixation_y  in enumerate(x.fixation_point_y) if (x.eye_movement_type[t] == 'Fixation')]), axis=1)



    return pd.Series(res)

def compute_segments_accelerometric_features_row(row):

    if row.has_gaze == 0:
        return pd.Series({f:np.nan for f in gaze_features['accelerometric_data']})
    
    res = {f:[] for f in gaze_features['accelerometric_data']}
    
    acc_list_tot = []
    # Segments bunds are in minutes and time in seconds 
    for idx_segment, (s, e) in enumerate(row.segments_bounds):
        
        #print(idx_segment, (s, e))
        acc_list = []

        for t, time in enumerate(row.time):
            
            if ((time / 60) >= s) and ((time / 60) <= e):
                
                ax = row.acceleration_x[t]
                ay = row.acceleration_y[t]
                az= row.acceleration_z[t]

                acc_list.append(np.sqrt(ax**2 + ay**2 + az**2))
        
        res['acceleration_norm_mean'].append(np.nanmean(acc_list) if len(acc_list) > 0 else np.nan)
        res['acceleration_norm_std'].append(np.nanstd(acc_list) if len(acc_list) > 0 else np.nan)

    res['acceleration_norm_mean_tot'] = np.nanmean( [ np.sqrt( ax**2 + ay**2 + az**2) for ax, ay, az in zip(row.acceleration_x, row.acceleration_y, row.acceleration_z)] )
    res['acceleration_norm_std_tot'] = np.nanstd( [ np.sqrt( ax**2 + ay**2 + az**2) for ax, ay, az in zip(row.acceleration_x, row.acceleration_y, row.acceleration_z)] )

    return pd.Series(res)

def compute_segments_gyroscopic_features_row(row):

    if row.has_gaze == 0:
        return pd.Series({f:np.nan for f in gaze_features['gyroscopic_data']})
    
    res = {f:[] for f in gaze_features['gyroscopic_data']}

    # Segments bunds are in minutes and time in seconds 
    for idx_segment, (s, e) in enumerate(row.segments_bounds):
        
        #print(idx_segment, (s, e))
        gyr_list = []

        for t, time in enumerate(row.time):
            
            if ((time / 60) >= s) and ((time / 60) <= e):
                
                gx = row.gyro_x[t]
                gy = row.gyro_y[t]
                gz = row.gyro_z[t]

                gyr_list.append(np.sqrt(gx**2 + gy**2 + gz**2))
        
        res['gyro_norm_mean'].append(np.nanmean(gyr_list) if len(gyr_list) > 0 else np.nan)
        res['gyro_norm_std'].append(np.nanstd(gyr_list) if len(gyr_list) > 0 else np.nan)

    res['gyro_norm_mean_tot'] = np.nanmean( [ np.sqrt( ax**2 + ay**2 + az**2) for ax, ay, az in zip(row.gyro_x, row.gyro_y, row.gyro_z)] )
    res['gyro_norm_std_tot'] = np.nanstd( [ np.sqrt( ax**2 + ay**2 + az**2) for ax, ay, az in zip(row.gyro_x, row.gyro_y, row.gyro_z)] )

    return pd.Series(res)

def compute_segments_gaze_features_row(row):

    
    if row.has_gaze == 0:
        return pd.Series({f:np.nan for f in gaze_features['gaze_data']})
    
    
    row['gaze_2D_path_length'] = [ np.sqrt( gx**2 + gy**2) for gx, gy in zip(np.ediff1d(row.gaze_2D_x), np.ediff1d(row.gaze_2D_y))]
    row['gaze_3D_path_length'] = [ np.sqrt( gx**2 + gy**2 + gz**2) for gx, gy, gz in zip(np.ediff1d(row.gaze_3D_x), np.ediff1d(row.gaze_3D_y), np.ediff1d(row.gaze_3D_z))]



    res = {f:[] for f in gaze_features['gaze_data']}
    
    # Keep same length as time
    row['gaze_2D_path_length'] = [0] + [ np.sqrt( gx**2 + gy**2) for gx, gy in zip(np.ediff1d(row.gaze_2D_x), np.ediff1d(row.gaze_2D_y))]
    row['gaze_3D_path_length'] =[0] +  [ np.sqrt( gx**2 + gy**2 + gz**2) for gx, gy, gz in zip(np.ediff1d(row.gaze_3D_x), np.ediff1d(row.gaze_3D_y), np.ediff1d(row.gaze_3D_z))]

    # Segments bunds are in minutes and time in seconds 
    for idx_segment, (s, e) in enumerate(row.segments_bounds):
        
        #print(idx_segment, (s, e))
        gaze_2D_list = []
        gaze_3D_list = []
        gaze_2D_length_path_list = []
        gaze_3D_length_path_list = []
        pupil_L_diameter_list = []
        pupil_R_diameter_list = []
        valid_R_list = []
        valid_L_list = []

        for t, time in enumerate(row.time):
            
            if ((time / 60) >= s) and ((time / 60) <= e):
                
                gaze_2D_x = row.gaze_2D_x[t]
                gaze_2D_y = row.gaze_2D_y[t]
                
                gaze_3D_x = row.gaze_3D_x[t]
                gaze_3D_y = row.gaze_3D_y[t]
                gaze_3D_z = row.gaze_3D_z[t]

                if pd.notna(gaze_2D_x) and pd.notna(gaze_2D_y):
                    #try:
                    gaze_2D_list.append(np.sqrt(gaze_2D_x**2 + gaze_2D_y**2))
                    #except:
                        #gaze_2D_list.append(np.nan)
                    #    print('Error in gaze_2D', gaze_2D_x, gaze_2D_y)
                    #    print("type:", type(gaze_2D_x), type(gaze_2D_y))
                
                if pd.notna(gaze_3D_x) and pd.notna(gaze_3D_y) and pd.notna(gaze_3D_z):
                    
                    #try:
                    gaze_3D_list.append(np.sqrt(gaze_3D_x**2 + gaze_3D_y**2 + gaze_3D_z**2))
                    #except:
                    #    print('Error in gaze_3D', gaze_3D_x, gaze_3D_y, gaze_3D_z)
                    #    print("type:", type(gaze_3D_x), type(gaze_3D_y), type(gaze_3D_z))
                        
                if pd.notna(row.gaze_2D_path_length[t]):
                    gaze_2D_length_path_list.append(row.gaze_2D_path_length[t])
                        
                if pd.notna(row.gaze_3D_path_length[t]):
                    gaze_3D_length_path_list.append(row.gaze_3D_path_length[t])
                    
                if pd.notna(row.pupil_L_diameter[t]):
                    pupil_L_diameter_list.append(row.pupil_L_diameter[t])
                if pd.notna(row.pupil_R_diameter[t]):
                    pupil_R_diameter_list.append(row.pupil_R_diameter[t])
                
                valid_L_list.append(row.valid_L[t])
                valid_R_list.append(row.valid_R[t])
                
                
        res['gaze_2D_norm_mean'].append(np.nanmean(gaze_2D_list) if len(gaze_2D_list) > 0 else np.nan)
        res['gaze_2D_norm_std'].append(np.nanstd(gaze_2D_list) if len(gaze_2D_list) > 0 else np.nan)
        
        res['gaze_3D_norm_mean'].append(np.nanmean(gaze_3D_list) if len(gaze_3D_list) > 0 else np.nan)
        res['gaze_3D_norm_std'].append(np.nanstd(gaze_3D_list) if len(gaze_3D_list) > 0 else np.nan)
        
        res['gaze_2D_path_length_mean'].append(np.nanmean(gaze_2D_length_path_list) if len(gaze_2D_length_path_list) > 0 else np.nan)
        res['gaze_2D_path_length_std'].append(np.nanstd(gaze_2D_length_path_list) if len(gaze_2D_length_path_list) > 0 else np.nan)
        
        res['gaze_3D_path_length_mean'].append(np.nanmean(gaze_3D_length_path_list) if len(gaze_3D_length_path_list) > 0 else np.nan)
        res['gaze_3D_path_length_std'].append(np.nanstd(gaze_3D_length_path_list) if len(gaze_3D_length_path_list) > 0 else np.nan)
        
        res['pupil_L_diameter_mean'].append(np.nanmean(pupil_L_diameter_list) if len(pupil_L_diameter_list) > 0 else np.nan)
        res['pupil_L_diameter_std'].append(np.nanstd(pupil_L_diameter_list) if len(pupil_L_diameter_list) > 0 else np.nan)

        res['pupil_R_diameter_mean'].append(np.nanmean(pupil_R_diameter_list) if len(pupil_R_diameter_list) > 0 else np.nan)
        res['pupil_R_diameter_std'].append(np.nanstd(pupil_R_diameter_list) if len(pupil_R_diameter_list) > 0 else np.nan)
        
        res['valid_L_prop'].append(np.nanmean(valid_L_list) if len(valid_L_list) > 0 else np.nan)
        res['valid_R_prop'].append(np.nanstd(valid_R_list) if len(valid_R_list) > 0 else np.nan)

    res['valid_R_prop_tot'] = np.nanmean(row.valid_R)
    res['valid_L_prop_tot'] = np.nanmean(row.valid_L)

    res['gaze_2D_path_length_mean_tot'] = np.nanmean(row.gaze_2D_path_length)
    res['gaze_2D_path_length_std_tot'] = np.nanstd(row.gaze_2D_path_length)
    
    res['gaze_3D_path_length_mean_tot'] = np.nanmean(row.gaze_3D_path_length)
    res['gaze_3D_path_length_std_tot'] = np.nanstd(row.gaze_3D_path_length)
    
    
    res['pupil_L_diameter_mean_tot'] = np.nanmean(row.pupil_L_diameter)
    res['pupil_L_diameter_std_tot'] = np.nanstd(row.pupil_L_diameter)
    res['pupil_R_diameter_mean_tot'] = np.nanmean(row.pupil_R_diameter)
    res['pupil_R_diameter_std_tot'] = np.nanstd(row.pupil_R_diameter)

    
    
    return pd.Series(res)


# Debugging snippet codes
# dset = get_dataset(dataset_name='base', verbose=False)

# df = dset.metadata.copy()

# df[df.trigram.isna()]

# from smartflat.utils.utils_io import fetch_has_gaze, get_data_root, load_df, save_df
# from smartflat.datasets.dataset_gaze import parse_tobii_data
# from smartflat.datasets.dataset_gaze import compute_segments_gaze_features, populate_gaze_data
# from smartflat.utils.utils_io import parse_participant_id
# df_merged[["task_number", "diag_number", "trigram", "date_folder"]] = df_merged["participant_id"].apply(parse_participant_id)



# df_merged['participant_id'].apply(lambda x: 'G150' in x).sum()


# df_merged['has_gaze_online']=df_merged.apply(fetch_has_gaze, axis=1, verbose=False)
# df_merged['has_gaze_online'].value_counts()

# r = select(df_merged, 'modality', 'Tobii')
# r = select(select(df_merged, 'modality', 'Tobii'), 'task_name', 'lego')

# r[r['has_gaze_online']==False].participant_id.unique()


# r = select(select(df_merged, 'task_name', 'lego'), 'modality', 'Tobii')
# for tn in r[r['has_gaze_online']==False].task_number.unique():
    
#     lp = glob(os.path.join('/diskA/sam_data/data-gaze/lego', f'*{tn}*'))

#     if len(lp) > 0:
#         yellow(tn)
        
#         print('\n'.join(lp))
        
#         if len(lp) == 1:
#             #yellow(tn)
#             gaze_path = lp[0]
#             #print(gaze_path)
#             #gdata = parse_tobii_data(gaze_path, data_type='gaze_data')

# df_merged[df_merged.participant_id.apply(lambda x: 'G150' in x)]

# test_g = df_merged[df_merged.participant_id.apply(lambda x: 'G150' in x)].copy()

# test_g['has_gaze'] = test_g.apply(fetch_has_gaze, axis=1, verbose=True)
# test_g['has_gaze']

# test_g



# test_g = compute_segments_gaze_features(test_g, verbose=False)

