import json
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from glob import glob
from json.decoder import JSONDecodeError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from smartflat.constants import (
    available_modality,
    available_tasks,
    delim,
    expected_folders,
    fix_dumjea_path,
    mapping_annotation_path_identifiers,
    mapping_boris_name_participant_id,
    mapping_incorrect_modality_name,
    mapping_participant_id_fix,
)
from smartflat.utils.utils import pairwise, predict_segments_from_embed_labels
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_io import get_data_root, parse_path


class AnnotationSmartflat(object):
    """
        Class handling the annotations from Vidat or Boris.
        
        
        Notes: 
            -   Time reference is always the one of the video (no sampling). 
            -   The start frames and stop frames always contains the video first and last frames.

    """

    def __init__(self,
                task_name,
                annotation_path=None,
                names = ['Ground Truth', 'Prediction (ours)'],
                overlap_list=[.1, .25, .5], 
                save_annotation_after_parsing=False, 
                verbose=False,
                annotation_parsing_method='use-aggregation-file' # 'use-aggregation-file' or 'per-file'
                ):
                    
        """ 
            We assume here that if the cpt are provided, they contain a [0] and [n_frame] values.
            If we provide start and end frames, we add [0] and [n_frames]
            software is 'boris' or 'vidat'
        """

        self.task_name = task_name
        self.annotation_path = annotation_path
        self.df = None
        self.is_parsed = False
        self.annotation_parsing_method = annotation_parsing_method
        self.save_annotation_after_parsing = False
        self.verbose = verbose
        
        
        if annotation_path is not None:
            self.has_annotation = True
            if annotation_path.endswith('.boris'):
                self.annotations_software = 'boris'
            elif annotation_path.endswith('.json'):
                self.annotations_software = 'vidat'
                    
        else:
            self.has_annotation = False
            self.annotations_software = None



    def parse(self, annot_all=None, participant_id=None, modality=None, verbose=False):
        
        if self.is_parsed:
            return
        
        if self.annotation_parsing_method == 'use-aggregation-file':
            
            if annot_all is None:
                
                annotation_path = os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all.csv')
                annot_all = pd.read_csv(annotation_path)
                
            self.df = annot_all[(annot_all['participant_id'] == participant_id) & (annot_all['modality'] == modality)]

            if len(self.df) == 0:
                if verbose:
                    pass#red(f'No annotations found for {participant_id}')
                self.has_annotation = False
                self.annotation_path = None
                
            else:
                if verbose:
                    green(f'Annotations found for {participant_id} - {modality} - { self.annotations_software}')
                self.n_frames_annot = self.df['End Frame'].max()
                self.df.assign(task_name=self.task_name, participant_id=participant_id)
                self.is_parsed = True
                self.has_annotation = True
                assert self.df.annotation_software.nunique() == 1
                self.annotation_software = self.df['annotation_software'].iloc[0]

        elif self.annotation_parsing_method == 'per-file':
            
            if self.has_annotation and (self.annotations_software == 'vidat'):
                #print(f'Parsing {self.annotation_path}')
                self.df = parse_vidat(self.task_name, self.annotation_path, save_df=self.save_annotation_after_parsing)
                
                self.n_frames_annot = self.df['End Frame'].max()
                self.n_frames_annot = self.df['End Frame'].max()
                self.df.assign(task_name=self.task_name, participant_id=participant_id)
                self.is_parsed = True
                
            elif self.has_annotation and (self.annotations_software == 'boris'):
                #print(f'Parsing {self.annotation_path}')
                self.df = parse_boris(self.task_name, self.annotation_path, save_df=self.save_annotation_after_parsing)
                                
                if self.df is None:
                    #print(f'/!\ No observations found for {self.annotation_path}')
                    self.has_annotation = False
                else:    
                    self.n_frames_annot = self.df['End Frame'].max()
                    self.df.assign(task_name=self.task_name, participant_id=participant_id)
                self.is_parsed = True
            else:
                self.is_parsed = False
                

def retrieve_sample_annotations(participant_id, annot_all=None):
    
    if annot_all is None:
        annotation_path = os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all.csv')
        annot_all = pd.read_csv(annotation_path)
    
    sample_annot = annot_all[annot_all['participant_id'] == participant_id]

    if len(sample_annot) == 0:
        red(f'No annotations found for {participant_id}')
        return None
    else:
        green(f'Annotations found for {participant_id}')
        return sample_annot
    
def process_annotation_folder(dset, root_dir, process=False):
    """
        Explore a folder with boris annotation files and:
            - If the participant is found in the present "gold" dataset, copy the boris file to their annotation folder.
            - Report in the file `clinicians_output_path` relevent information when the administration is not found. 
            - Parse all annotation projects (all potential ones within boris files) and concatenate them to save all annotations in 
                the `annotation_output_path` file.
    
    """
    
    registered_participant_ids = dset.metadata.participant_id.unique()


    output_annot_dir = os.path.join(os.path.dirname(get_data_root()), 'data-annotations')
    os.makedirs(output_annot_dir, exist_ok=True)
    output_data_dir = get_data_root()

    today_date = datetime.today().strftime('%d%m%Y')
    os.makedirs(os.path.join(get_data_root(), 'dataframes', 'annotations'), exist_ok=True)
    clinicians_output_path = os.path.join(get_data_root(), 'dataframes', 'annotations', f'missing_annotation_mapping_{today_date}.csv')
    
    annotation_output_path = os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all.csv')
    annotation_output_path_dated = os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all_{today_date}.csv')

    commands = []

    boris_df_list = []; vidat_df_list = []
    missing_annotation_mapping = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".boris"):
                boris_path = os.path.join(root, file)
                boris_df_list, commands, missing_annotation_mapping = helper_parse_boris(output_data_dir, boris_path, boris_df_list, commands, missing_annotation_mapping, registered_participant_ids)
                
                commands.append(["cp", boris_path, output_annot_dir])
                
                continue
            
            elif file.endswith('.json'):
                
                if file in ['._DUMJea2018.json', 'vidat_config_final_november2022.json', 'config_final.json', '._config_final.json', '._config_smartflat.json', 'config_smartflat.json']:
                    continue
                vidat_path = os.path.join(root, file)
                vidat_df_list, commands, missing_annotation_mapping = helper_parse_vidat(output_data_dir, vidat_path, vidat_df_list, commands, missing_annotation_mapping, registered_participant_ids)
                        
                commands.append(["cp", vidat_path, output_annot_dir])

                        
    df_boris = pd.concat(boris_df_list)
    df_vidat = pd.concat(vidat_df_list)

    annot_df = pd.concat([df_boris, df_vidat])
    
    # Corrections
    annot_df.loc[annot_df['label'] == 'E1', 'Categorie'] = 'A'
    annot_df.loc[annot_df['label'] == 'E1', 'Categorie Label'] = 'Etapes de la tache'
    annot_df.loc[annot_df['label'] == 'E1', 'label'] = 'A10'

    missing_annotation_mapping_df = pd.DataFrame(missing_annotation_mapping).drop_duplicates(['annotation_path'])

    if process:
        
        missing_annotation_mapping_df.to_csv(clinicians_output_path, index=False)
        print(f"Unknown annotations reported in {clinicians_output_path}")

        display(annot_df.drop_duplicates(['participant_id', 'annotation_software']).annotation_software.value_counts())
        
        annot_df.to_csv(annotation_output_path_dated, index=False)
        annot_df.to_csv(annotation_output_path, index=False)
        red(f"Save all-annotations file in {annotation_output_path}")
        annot_df.to_csv(os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all_raw.csv'), index=False)


        # **Extra filtering based on the basename**

        annotation_output_path = os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all_raw.csv')
        annot_df = pd.read_csv(annotation_output_path)
        annot_df.loc[annot_df['annotation_software'] == 'vidat', 'basename'] = 'placeholder_vidat'
        annot_df['basename_annotation_path'] = annot_df['annotation_path'].apply(os.path.basename)
        annot_df["identifier"] = annot_df.apply(lambda x: "{}_{}_{}_{}".format(x.participant_id, x.task_name, x.modality, x.video_name),axis=1,)
        annot_df['n_annot_path'] = annot_df.groupby(['annotation_software', 'task_name', 'identifier']).basename_annotation_path.transform(lambda x: len(np.unique(x)))
        annot_df.drop_duplicates(subset=['task_name', 'identifier', 'basename_annotation_path', 'Scene Number'], inplace=True)
        n = len(annot_df); print(f'N={annot_df.participant_id.nunique()} identifier and {n} annotations')

        # Remove duplicate path projects  by using the ones with the maximum number of annotation 
        mannotdf = annot_df[annot_df['n_annot_path'] > 1]
        print(f"Initial number of annotations: {n}")
        print(f"Number of unique identifiers with multiple paths: {mannotdf['identifier'].nunique()}")
        mannotdf.groupby(['identifier']).annotation_path.value_counts().to_frame()
        annot_comparison_size_per_basename = mannotdf.groupby(['identifier', 'basename_annotation_path'])['annotation_path'].agg('count').reset_index().rename(columns={'annotation_path':'n_annot_path'})
        max_annot_mapping = annot_comparison_size_per_basename.sort_values(['identifier', 'n_annot_path'], ascending=False).drop_duplicates(['identifier'], keep='first')[['identifier', 'basename_annotation_path']].set_index('identifier').to_dict()['basename_annotation_path']
        annot_df['annotation_path_to_use'] = annot_df.apply(lambda x: max_annot_mapping[x.identifier] if x.identifier in max_annot_mapping.keys() else x.basename_annotation_path, axis=1)

        annot_df = annot_df[annot_df['annotation_path_to_use'] == annot_df['basename_annotation_path']]
        print(f'Removed {n - len(annot_df)} duplicates')
        n = len(annot_df); print(f'N={annot_df.participant_id.nunique()} identifier and {n} annotations')

        # Remove duplicates basename projects  by using the ones with the maximum number of annotation 
        annot_df['n_basename'] = annot_df.groupby(['annotation_software', 'task_name', 'identifier']).basename.transform(lambda x: len(np.unique(x)))
        annot_df.loc[annot_df['annotation_software'] == 'vidat','n_basename'] = 1
        display(annot_df['n_basename'].value_counts())

        mannot_df = annot_df[annot_df['n_basename'] > 1].sort_values('identifier')
        display(mannot_df.groupby(['task_name', 'participant_id', 'annotation_software', 'annotation_path', 'basename']).size())
        display(mannot_df.groupby(['identifier' ,'basename']).size().reset_index().rename(columns={0:'n'}).sort_values(['identifier', 'n'], ascending=False))
        max_annot_mapping = mannot_df.groupby(['identifier' ,'basename']).size().reset_index().rename(columns={0:'n'}).sort_values(['identifier', 'n'], ascending=False).drop_duplicates('identifier', keep='first').set_index('identifier').to_dict()['basename']
        annot_df['basename_to_use'] = annot_df.apply(lambda x: max_annot_mapping[x.identifier] if x.identifier in max_annot_mapping.keys() else x.basename, axis=1)
        annot_df = annot_df[annot_df['basename'] == annot_df['basename_to_use']]
        print(f'Removed {n - len(annot_df)} duplicate basename  projects')
        n = len(annot_df); print(f'N={annot_df.participant_id.nunique()} identifier and {n} annotations')

        annot_to_discrard_path = ['/Volumes/Smartflat/data-annotations/CHAAli_lego2022_SDS2_T0_FB_incomplet GP2.boris',
                                '/Volumes/Smartflat/data-annotations/023_IM_SDS2_T0_gateau.boris',
                                '/Volumes/Smartflat/data-annotations/SDS2_Y0_gateau_024-MF_FB_manque fin.boris',
                                '023_ILIMil_SDS2_M12_Y2_lego_2024_FB.boris',
                                '/Volumes/Smartflat/data-annotations/023_ILIMil_SDS2_M12_Y2_lego_2024_FB.boris'
                                ]

        n = len(annot_df)
        annot_df = annot_df[~annot_df['annotation_path'].isin(annot_to_discrard_path)]
        print(f'Removed {n - len(annot_df)} manually removed annotations')

        annot_df.to_csv(os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotations_all.csv'), index=False)
        print(f'Saved {n} annotations to {os.path.join(get_data_root(), "dataframes", "annotations", "annotations_all.csv")}')


        
        for command in commands:
            blue(" ".join(command))
            subprocess.run(command)

    else:
        for command in commands:
            print(" ".join(command))
            
            
            
    return annot_df, missing_annotation_mapping_df, commands

def helper_parse_vidat(output_dir, vidat_path, df_list, commands, missing_annotation_mapping, registered_participant_ids):
    
    # parse annotation sample        
    annot_df = parse_vidat(task_name='cuisine', annotation_path=vidat_path, save_df=False)
    
    if os.path.basename(vidat_path) in mapping_annotation_path_identifiers.keys():
        video_name, modality, participant_id, task_name = mapping_annotation_path_identifiers[os.path.basename(vidat_path)]
    
    
            
    
    
    if participant_id not in registered_participant_ids:
        red(f"/!\ Participant {participant_id} not registered/found. Exit")
        missing_annotation_mapping.append({'annotation_path': vidat_path, 
                                           'task_name': task_name,
                                        'example_video_path': np.nan,
                                        'project_name': np.nan,
                                        'project_date': np.nan,
                                        'participant_id': 'A DETERMINER'
                                        })   
        # import subprocess
        # subprocess.run(['cp', vidat_path, '/Volumes/Smartflat/data-gold-final/dataframes/annotations/annotations-sans-participants/'])


    if annot_df is not None:
        df_list.append(annot_df.assign(annotation_path=vidat_path, 
                                       task_name=task_name, 
                                        participant_id=participant_id, 
                                        modality=modality, 
                                        video_name = video_name, 
                                        basename=np.nan, 
                                        annotation_software='vidat'
                                        ))



    output_dir_participant = os.path.join(output_dir, task_name, participant_id, "Annotation")

    assert os.path.exists(
        output_dir_participant
    ), f"Output directory {output_dir_participant} does not exist"


    commands.append(["cp", vidat_path, output_dir_participant])
    
    return df_list, commands, missing_annotation_mapping

def helper_parse_boris(output_dir, boris_path, df_list, commands, missing_annotation_mapping, registered_participant_ids):
    
    report_text = f"{delim}\nFound boris file: {boris_path}"

    # Get data
    try:
        with open(boris_path, "r") as f:
            data = json.load(f)
    except (JSONDecodeError, UnicodeDecodeError):
        red(f"Corrupted file: {boris_path}")
        return df_list, commands, missing_annotation_mapping


    # if len(data["observations"].keys()) > 1:
    #     red(f"Multiple observations files found for {boris_path}.")
    #     #print(data["observations"].keys())
    #     #print(report_text)
    #     #yellow('{} : {},'.format(boris_path, list(data['observations'].keys())))
    #     pass


    basenames = list(data["observations"].keys())
    
    for basename in basenames:
        
        # 1) Parsing task_name, participant_id, modality, video_name
        
        annot_video_path = data["observations"][basename]["file"]["1"][0]
        
        if os.path.basename(boris_path) in mapping_annotation_path_identifiers.keys():
            video_name, modality, participant_id, task = mapping_annotation_path_identifiers[os.path.basename(boris_path)]
            
            
        else:
            video_name, modality, participant_id, task = parse_path(annot_video_path) 

    
        if 'from' in annot_video_path: 
            continue
        elif 'stagiaire' in annot_video_path:
            continue
        elif 'vrac' in boris_path:
            continue
        
        elif 'noémie' in boris_path:
            continue
    

        # 2) Checks
        if modality not in expected_folders:
            
            if modality in mapping_incorrect_modality_name.keys():
                green(f'Found {modality} -> {mapping_incorrect_modality_name[modality]} ({annot_video_path}) ({os.path.basename(boris_path)})')
                modality = mapping_incorrect_modality_name[modality]
            
            elif os.path.basename(boris_path) in mapping_annotation_path_identifiers.keys():
                video_name, modality, participant_id, task = mapping_annotation_path_identifiers[os.path.basename(boris_path)]
                
            else:
                red(f'Aborted as modality={modality} - {boris_path} - {annot_video_path}.') #TODO: tmeporary                        
                t_duration = np.sum(list(data["observations"][basename]['media_info']['length'].values())) / 60


                missing_annotation_mapping.append({'annotation_path': boris_path, 
                                                'example_video_path': annot_video_path, 
                                                'project_name': data['project_name'],
                                                'project_date': data['project_date'],
                                                'total_duration': t_duration,
                                                'participant_id': 'A DETERMINER'
                                                })     
                
                # import subprocess
                # subprocess.run(['cp', boris_path, '/Volumes/Smartflat/data-gold-final/dataframes/annotations/annotations-sans-participants/'])
            
                continue
                
        if participant_id in mapping_participant_id_fix:
            green(
                f"Fixed participant id: {participant_id} to {mapping_participant_id_fix[participant_id]}"
            )
            participant_id = mapping_participant_id_fix[participant_id]
            
        elif participant_id in mapping_boris_name_participant_id.keys():
            
            participant_id = mapping_boris_name_participant_id[participant_id]
        
            
        if "lego" in task.lower():
            task = "lego"
        elif "gateau" in task.lower() or 'cuisine' in task.lower() or 'Cuisine_GP' in task.lower():
            task = "cuisine"
        else:
            print(boris_path, video_name, modality, participant_id, task, annot_video_path)
            raise ValueError(f"Unknown task: {task}") #to be added to api.constants: 
        
        
        print(f"{video_name:<20}{modality:<15}{participant_id:<50}{task:<80}")

        # print(f'Participant: {participant_id}, task: {task}, modality: {modality}, video: {video_name}')
        if participant_id not in registered_participant_ids:
            red(f"/!\ Participant {participant_id} not registered/found. Exit")
            t_duration = np.sum(list(data["observations"][basename]['media_info']['length'].values())) / 60

            missing_annotation_mapping.append({'annotation_path': boris_path, 
                            'example_video_path': annot_video_path, 
                            'project_name': data['project_name'],
                            'project_date': data['project_date'],
                            'total_duration': t_duration,
                            'participant_id': 'A DETERMINER'
                            })   
            
            # import subprocess
            # subprocess.run(['cp', boris_path, '/Volumes/Smartflat/data-gold-final/dataframes/annotations/annotations-sans-participants/'])
            

        elif not os.path.exists(os.path.join(output_dir, task, participant_id)):
            red(f"Path {os.path.join(output_dir, task, participant_id)} don't exist.")
            print(boris_path)
            
        else:
            #green(f"Found participant {participant_id} in the dataset")

            output_dir_participant = os.path.join(output_dir, task, participant_id, "Annotation")

            assert os.path.exists(
                output_dir_participant
            ), f"Output directory {output_dir_participant} does not exist"


            commands.append(["cp", boris_path, output_dir_participant])   
            
        # parse annotation sample
        annot_df = parse_boris(task, boris_path, basename=basename, save_df=False)
        if annot_df is not None:
            
            t_duration = np.sum(list(data["observations"][basename]['media_info']['length'].values())) / 60
            
            fps = list(data["observations"][basename]['media_info']['fps'].values())[0]
            
            
            df_list.append(annot_df.assign(annotation_path=boris_path, 
                                           task_name=task,
                                            participant_id=participant_id, 
                                            modality=modality, 
                                            video_name=video_name,
                                            basename=basename, 
                                            annotation_software='boris',
                                            fps=fps,
                                            t_duration=t_duration
                                            ))
    
    return df_list, commands, missing_annotation_mapping
       
def parse_vidat(task_name, annotation_path, save_df=True):
    
    print(f'Parsing {annotation_path}')
    
    with open(annotation_path, 'r') as f:
        data = json.load(f)
        
    mapping_action_vidat = {}
    for dict_action in data['config']['actionLabelData']:

        mapping_action_vidat[dict_action['id']] = {'label': dict_action['name'],
                                                   'color': dict_action['color']
                                                  }
    
    order_categories, cuisine_mapping_dict, lego_mapping_dict, mapping_categorie_track, mapping_boris_vidat  = get_annotation_constants()

    if task_name == 'cuisine':
        mapping_dict = cuisine_mapping_dict 
    elif task_name == 'lego':
        mapping_dict = lego_mapping_dict
    else:
        raise ValueError
        
    fps = data['annotation']['video']['fps']
    n_frames = data['annotation']['video']['frames']

    df = pd.DataFrame(columns=['Scene Number', 'Start Frame', 'Start Time (seconds)', 'End Frame', 'End Time (seconds)', 'Length (frames)', 'Length (seconds)', 'fps', 't_duration', 'n_frames'])

    for i, annot_dict in enumerate(data['annotation']['actionAnnotationList']):
        new_row = pd.DataFrame({
            'Scene Number': i,
            'Start Frame': np.round(annot_dict['start'] * fps).astype(int),
            'Start': annot_dict['start'],
            'End Frame': np.round(annot_dict['end'] * fps).astype(int),
            'End': annot_dict['end'],
            'Length (frames)': np.round((annot_dict['end'] - annot_dict['start']) * fps).astype(int),
            'Length': (annot_dict['end'] - annot_dict['start']),
            'Color': int(mapping_dict[mapping_action_vidat[annot_dict['action']]['label']]['color']),
            'label': mapping_action_vidat[annot_dict['action']]['label'],
            'Semantic': mapping_dict[mapping_action_vidat[annot_dict['action']]['label']]['semantic'],
            'code': int(mapping_dict[mapping_action_vidat[annot_dict['action']]['label']]['code']),
            'Categorie': mapping_action_vidat[annot_dict['action']]['label'][0],
            'Categorie Label': mapping_categorie_track[mapping_action_vidat[annot_dict['action']]['label'][0]],
            'Description': annot_dict['description'] if len(annot_dict['description']) > 0 else "None",
            'type': 'interval',
            'fps': fps,
            't_duration': (n_frames / fps) / 60,
            'n_frames': n_frames
            
        }, index=[i])
        
        df = pd.concat([df, new_row], ignore_index=True)

    df['Length (s)'] = (df['End'] - df['Start']).round(2)
    df['Start'] = df['Start'].apply(lambda x: datetime.fromtimestamp(datetime(2022,1,1).timestamp()+x))
    df['End'] = df['End'].apply(lambda x: datetime.fromtimestamp(datetime(2022,1,1).timestamp()+x))

    #df.drop_duplicates(subset=['Start Frame', 'End Frame'], inplace=True)
    
    # TODOCHECK
    to_duplicate_row = df[df['label']=='A2'].copy()
    to_duplicate_row['label'] = 'C1'
    to_duplicate_row['Color'] = 0
    to_duplicate_row['Categorie'] = 'C'
    to_duplicate_row['Categorie Label'] = "Prise d'information visuelle"
    to_duplicate_row['code'] = 25
    df = pd.concat([df, to_duplicate_row], ignore_index=True)


    df.sort_values(by='Start Frame', inplace=True)
    
    df = fix_annotation_vidat(df, annotation_path)
    # Fix compared to the boris-vidat combiantion
    df.loc[(df['code'] == 31) & (df['label'] == 'A10'), 'code'] = 47

    
    if save_df:
        output_path = annotation_path.split('.')[0] + '.csv'
        df.to_csv(output_path, index=False)

    return df

def parse_boris(task_name, annotation_path, basename=None, save_df=True, verbose=False):
    """Parse Boris file to render the dataframe.
    
    Note: 
        For e.g. annotations/mail-flavie/OLIFra_SC_070618_gateau_FB.boris:
        data['observations'].keys() -> ['PERSyl_SC_250918_gateau_FB', 'OLIFra_SC_070618_gateau_FB']
        
        While it seems the key is the file name, files might host multiple annotations...
    """
    
    
    order_categories, cuisine_mapping_dict, lego_mapping_dict, mapping_categorie_track, mapping_boris_vidat = get_annotation_constants()
    
    
    if task_name == 'cuisine':
        mapping_dict = cuisine_mapping_dict 
    elif task_name == 'lego':
        mapping_dict = lego_mapping_dict
    else:
        raise ValueError
    
    # Get data
    with open(annotation_path, 'r') as f:
        data = json.load(f)
        
    if len(data['observations'].keys()) > 1 and basename is None:
        red(f"Multiple observations files found for {annotation_path}.")
        print(data['observations'].keys())
        
            
    #basename_index = disambiguation_observations_boris_files_.get(os.path.basename(annotation_path), 0)
    #if VERBOSE and basename_index != 0: 
    #pass#print(f'[annotation retrieval] Use disambiguation for {annotation_path}')
    #basename = list(data['observations'].keys())[basename_index]
    if basename is None:
        #red('Annotation name not provided (basename). First is used by default.')
        basename = list(data['observations'].keys())[0]
        
    
    if len(data['observations'][basename]['events']) == 0:
        red("No observations found in the file.") if verbose else None
        return None
        

    text = "\nObservations files: {}\nBasename: {}\nFile:{}}}".format(annotation_path, basename, data['observations'].keys())
    if verbose: 
        pass#print(text)
        
    annot_standard_df = pd.DataFrame(data['behaviors_conf']).T
    if len(annot_standard_df) >=26:
        
        output_p = os.path.join(get_data_root(), 'dataframes', 'annotations', f'annot_standard_boris_df_{len(annot_standard_df)}.csv')
        annot_standard_df.to_csv(output_p)
        print(f'Saved {len(annot_standard_df)} annotations to {output_p}')

    # Parse fps 
    fps = list(data['observations'][basename]['media_info']['fps'].values())[0]
    if np.std(list(data['observations'][basename]['media_info']['fps'].values())) > 1e-7 :
        raise Exception
    length_seconds =  list(data['observations'][basename]['media_info']['length'].values())[0] 
    
    # 
    if len(data['observations'][basename]['events'][0]) == 5:
        for (_, _, _, d, _) in data['observations'][basename]['events']:

            assert d == ""
            
        # Parse observation data
        obsdf = pd.DataFrame({'start_time': [start_time for (start_time, _ , _, _, _) in data['observations'][basename]['events']],
                            'category': [category for (_, _ , category, _, _) in data['observations'][basename]['events']], 
                            'description': [description for (_, _ , _, _, description) in data['observations'][basename]['events']], 
                            })
    elif len(data['observations'][basename]['events'][0]) == 6:
        for (_, _, _, d, _, _) in data['observations'][basename]['events']:
            assert d == ""

        # Parse observation data
        obsdf = pd.DataFrame({'start_time': [start_time for (start_time, _ , _, _, _, _) in data['observations'][basename]['events']],
                            'category': [category for (_, _ , category, _, _, _) in data['observations'][basename]['events']], 
                            'description': [description for (_, _ , _, _, description, _) in data['observations'][basename]['events']], 
                            })

    # Create annotation dataframe
    df = pd.DataFrame(columns=['Scene Number', 'Start Frame', 'Start', 'End Frame', 'End', 'Length (frames)',
                               'Length', 'Color', 'label', 'Semantic', 'code', 'Categorie', 'Categorie Label', 'Description', 'type', 'fps', 'n_frames_max'])

    for (obs_type, category) in annot_standard_df[['type', 'code']].to_numpy():

        # Get the observations performed in this category
        cat_obsdf = obsdf[obsdf['category'] == category].assign(obs_type=obs_type)

        # Observation is a segment defined by two entries
        if obs_type == 'State event':
            #print(f'-> State event: {category}')    

            for i, ((start_time, description), (end_time, _))  in enumerate(pairwise(cat_obsdf[['start_time', 'description']].to_numpy())):
                
                if i % 2 == 1: 
                    continue

                try:
                    df_row = pd.DataFrame({'Scene Number': np.nan, # Filled afterward
                                            'Start Frame': np.round(start_time*fps).astype(int), 
                                            'Start': start_time, 
                                            'End Frame': np.round(end_time*fps).astype(int), 
                                            'End': end_time, 
                                            'Length (frames)': np.round((end_time - start_time)*fps).astype(int), 
                                            'Length': (end_time - start_time), 
                                            'Color': int(mapping_dict[mapping_boris_vidat[category]]['color']), 
                                            'label': mapping_boris_vidat[category],
                                            'Semantic': mapping_dict[mapping_boris_vidat[category]]['semantic'],
                                            'code': int(mapping_dict[mapping_boris_vidat[category]]['code']),
                                            'Categorie':  mapping_boris_vidat[category][0],
                                            'Categorie Label': mapping_categorie_track[mapping_boris_vidat[category][0]],
                                            'Description': description if len(description) > 0 else "None",
                                            'type': 'interval',
                                            'fps': fps,
                                            }, index=[0])
                    df = pd.concat([df, df_row], ignore_index=True)

                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print(text)
                    red(f'-> this annotation file may be associated with the wrong task_name={task_name}')
                    print(text)
                    print('continue')
                    
                    df_row = pd.DataFrame({'Scene Number': np.nan, # Filled afterward
                        'Start Frame': np.round(start_time*fps).astype(int), 
                        'Start': start_time, 
                        'End Frame': np.round(end_time*fps).astype(int), 
                        'End': end_time, 
                        'Length (frames)': np.round((end_time - start_time)*fps).astype(int), 
                        'Length': (end_time - start_time), 
                        'Color': np.nan, #int(mapping_dict[mapping_boris_vidat[category]]['color']), 
                        'label': mapping_boris_vidat[category],
                        'Semantic': np.nan, #mapping_dict[mapping_boris_vidat[category]]['semantic'],
                        'code': np.nan, #int(mapping_dict[mapping_boris_vidat[category]]['code']),
                        'Categorie':  mapping_boris_vidat[category][0],
                        'Categorie Label': mapping_categorie_track[mapping_boris_vidat[category][0]],
                        'Description': description if len(description) > 0 else "None",
                        'type': 'interval',
                        'fps': fps
                        }, index=[0])
                    df = pd.concat([df, df_row], ignore_index=True)

                                        
                #    print(text)
                #    red(f'-> this annotation file may be associated with the wrong task_name={task_name}')

        # Observation is a point event
        elif obs_type == 'Point event':
            #print(f'-> Point event: {category}')    
            #if len(cat_obsdf) > 0 and cat_obsdf['category'].iloc[0] == 'E.N. - Négligence du Contexte':
            #    pass
                #print("Négligence du Contexte")
                #display(cat_obsdf)

            if len(cat_obsdf) > 0:

                for start_time, description  in cat_obsdf[['start_time', 'description']].to_numpy():

                    #try:
                    df_row = pd.DataFrame({'Scene Number': np.nan, # Filled afterward
                                            'Start Frame': np.round(start_time*fps).astype(int), 
                                            'Start': start_time, 
                                            'End Frame': np.nan, 
                                            'End':  np.nan, 
                                            'Length (frames)':  np.nan, 
                                            'Length':  np.nan,
                                            'Color': int(mapping_dict[mapping_boris_vidat[category]]['color']), 
                                            'label': mapping_boris_vidat[category],
                                            'Semantic': mapping_dict[mapping_boris_vidat[category]]['semantic'],
                                            'code': int(mapping_dict[mapping_boris_vidat[category]]['code']),
                                            'Categorie':  mapping_boris_vidat[category][0],
                                            'Categorie Label': mapping_categorie_track[mapping_boris_vidat[category][0]],
                                            'Description': description if len(description) > 0 else "None",
                                            'type': 'point',
                                            'fps': fps
                                            }, index=[0])
                    df = pd.concat([df, df_row], ignore_index=True)
                    # except Exception as e:
                    #     print(e)
                    #     traceback.print_exc()
                    #     print(text)
                        
                    #     red(f'-> this annotation file may be associated with the wrong task_name={task_name}')
                    #     print('continue')
                        
                    #     df_row = pd.DataFrame({'Scene Number': np.nan, # Filled afterward
                    #         'Start Frame': np.round(start_time*fps).astype(int), 
                    #         'Start': start_time, 
                    #         'End Frame': np.nan, 
                    #         'End': np.nan, 
                    #         'Length (frames)': np.nan,
                    #         'Length': np.nan,
                    #         'Color': np.nan, #int(mapping_dict[mapping_boris_vidat[category]]['color']), 
                    #         'label': mapping_boris_vidat[category],
                    #         'Semantic': np.nan, #mapping_dict[mapping_boris_vidat[category]]['semantic'],
                    #         'code': np.nan, #int(mapping_dict[mapping_boris_vidat[category]]['code']),
                    #         'Categorie':  mapping_boris_vidat[category][0],
                    #         'Categorie Label': mapping_categorie_track[mapping_boris_vidat[category][0]],
                    #         'Description': description if len(description) > 0 else "None",
                    #         'Type': 'point',
                    #         'fps': fp
                    #         }, index=[0])
                    #     df = pd.concat([df, df_row], ignore_index=True)

        else:
            raise ValueError


    df['Length (s)'] = (df['End'] - df['Start']).round(2)
    df['Start'] = df['Start'].apply(lambda x: datetime.fromtimestamp(datetime(2022,1,1).timestamp()+x))
    df['End'] = df['End'].apply(lambda x: datetime.fromtimestamp(datetime(2022,1,1).timestamp()+x) if not np.isnan(x) else x)

    #df.drop_duplicates(subset=['Start Frame', 'End Frame'], inplace=True)

    # Set the first step of looking at the recipe as a regular visual inspection interval
    to_duplicate_row = df[df['label']=='A2'].copy()
    to_duplicate_row['label'] = 'C1'
    to_duplicate_row['Color'] = 0
    to_duplicate_row['Categorie'] = 'C'
    to_duplicate_row['Categorie Label'] = "Prise d'information visuelle"
    to_duplicate_row['code'] = 25
    df = pd.concat([df, to_duplicate_row], ignore_index=True)


    df.sort_values(by='Start Frame', inplace=True)
    df['Scene Number'] = df.reset_index(drop=True).index + 1


    if save_df:
        output_path = annotation_path.split('.')[0] + '.csv'
        df.to_csv(output_path, index=False)
        print(f'Saved {output_path}')

    return df
     
def fix_annotation_vidat(df, annotation_path):
    
    if annotation_path.split('/')[-3] == 'DUMJEA':
        print("Correcting DUMJEA..")
        
        offset_dumjean_seconds = 140
        offset_dumjean_frames = 3500
        df['Start Frame'] = df['Start Frame'] + offset_dumjean_frames
        df['End Frame'] = df['End Frame'] + offset_dumjean_frames
        df['Start'] = df['Start'] + timedelta(0, offset_dumjean_seconds)
        df['End'] = df['End'] + timedelta(0, offset_dumjean_seconds)

        if os.path.isfile(fix_dumjea_path):
            df_fix = pd.read_csv(fix_dumjea_path, sep=';')
            df = pd.concat([df, df_fix], ignore_index=True)
            
        df.sort_values(by='Start Frame', inplace=True)
    return df 

def global_annotation_file_from_dset(dset, save_df=True):
    """ Create a global annotation file from a dataset object.
    Use row.annotations.df and assign participant_id, task_name, annotation_path.
    
    Example:
    
        ldf = global_annotation_file(dset, save_df=True)

    """
    g_columns = ['task_name', 'participant_id', 'annotation_path', 'type', 'Categorie', 'label', 'Semantic', 
                'Categorie Label', 'Length (frames)', 'Start Frame', 'End Frame', 'Length (s)', 
                'Start',  'End', 'Color', 'code',  'Description']
    annot_col_renaming = {'Length (frames)': 'duration', 'Start Frame':'start_frame', 'End Frame': 'end_frame'}

    # Parse annotations
    dset.parse_annotations()
    
    
    # Concatenate and post-process annotations entries
    global_annotations = pd.concat([annot.df.assign(task_name=tn, participant_id=pid, annotation_path=annot.annotation_path) for annot, tn, pid in df.drop_duplicates(['task_name', 'participant_id'])[['annotation', 'task_name', 'participant_id']].itertuples(index=False) if annot.has_annotation])
    ldf = global_annotations[g_columns].rename(columns=annot_col_renaming).sort_values(['task_name', 'participant_id', 'Categorie', 'Semantic'])
    ldf.drop_duplicates(['task_name', 'participant_id', 'Semantic', 'start_frame', 'end_frame'], inplace=True)

    os.makedirs(os.path.join(get_data_root(), 'dataframes', 'annotations'), exist_ok=True)


    if save_df:
    
        today_date = datetime.today().strftime('%d%m%Y')
        ldf.to_csv(os.path.join(get_data_root(), 'dataframes', 'annotations', f'annotation_all_{today_date}.csv'))
        
    return ldf

def sync_annotations(process=True, machine_name="pomme"):
    remote_output_dir = get_data_root(machine_name=machine_name)

    commands = []
    for path in glob(os.path.join(get_data_root("Smartflat"), "*", "*", "Annotation")):
        task_name, participant_id = path.split("/")[-3:-1]
        commands.append(
            [
                "rsync",
                "-ahuvz",
                "--progress",
                path,
                f"{machine_name}:{os.path.join(remote_output_dir, task_name, participant_id)}",
            ]
        )

    if process:
        for command in commands:
            blue(" ".join(command))
            subprocess.run(command)
    else:
        for command in commands:
            yellow(" ".join(command))
            
def plot_chronogram(df, title=None, annotation_path=None):
        
    order_categories, cuisine_mapping_dict, lego_mapping_dict, mapping_categorie_track, mapping_boris_vidat  = get_annotation_constants()

    task_names = df.task_name.unique()
    
    if len(task_names) > 1: 
        raise ValueError("Multiple tasks found in the dataframe")
    
    else:
        task_name = task_names[0]
        
        
    if title is None:
        
        title = f"Chronogram {df.participant_id.iloc[0]} ({task_name}):"
        
    if task_name == 'cuisine':
        mapping_dict = cuisine_mapping_dict 
    elif task_name == 'lego':
        mapping_dict = lego_mapping_dict
    else:
        raise ValueError
    
    
    fig = px.timeline(df, x_start="Start", x_end="End", y="Categorie Label", 
                      category_orders = {'Categorie Label': ['Etapes de la tache', 
                                                             "Touche d'un object/ingredient", 
                                                             "Prise d'information visuelle", 
                                                             'Actions sans buts', 
                                                             'Evenements', 
                                                             'Langages', 
                                                             'Erreurs Comportementales', 
                                                             'Erreurs Neuropsychologiques',
                                                             'Autre'],
                                        'Semantic': order_categories}, 
                      title=title,
                      #animation_group='Categorie Label',
                      color_discrete_sequence=px.colors.qualitative.Plotly,
                      hover_name='Semantic', hover_data={'label':True, 'Start':True, 'End': True, 'Semantic':True, 'Length (s)':True, 'Description':True}, 
                      color='Semantic', height=900, width=1500)

    #fig.update_yaxes(autorange="reversed") 
    #fig.layout.xaxis.type = 'linear'
    #fig.data[0].x = df.Length.tolist()
    fig.show()
    
    if annotation_path is not None:
        with open(annotation_path.replace('json', 'html').replace('boris', 'html'), 'a') as f:
            f.write(fig.to_html(full_html=False))

    return 

def create_label_array_from_dataframe(adf, category='A', N=None, verbose=True):
    
    _, _, _, mapping_categorie_track, _  = get_annotation_constants()

    adf.rename(columns={'Start Frame': 'start_frame', 'End Frame': 'end_frame'}, inplace=True)
    #adf.start_frame = adf.start_frame.astype(int)
    #adf.end_frame = adf.end_frame.astype(int) # point evets don't have end frame
    adf['n_frames_max'] = adf.apply(lambda x: int((x.t_duration * 60) * x.fps), axis=1)
    adf['start_percentile'] = adf.apply(lambda x: x['start_frame'] / x['n_frames_max'], axis=1)
    adf['end_percentile'] = adf.apply(lambda x: x['end_frame'] / x['n_frames_max'], axis=1)

    if N is None:
        y = np.full(adf['n_frames_max'].iloc[0], -1)  # Fill with a default label, e.g., 'O'
    else:
        N = int(N)
        y = np.full(N, -1)
        
    for _, row in adf[adf['Categorie'] == category].iterrows():
        
        if row['type'] == 'interval':
            
            if verbose:
                segment = y[row['start_frame']:int(row['end_frame'])]
                if np.all(segment == -1):
                    pass#print("Segment is untouched")
                else:
                    print(f'\nFrom {row["start_frame"]} to {row["end_frame"]}: {row["label"]} - {row["Semantic"]}')
                    unique, counts = np.unique(segment[segment != -1], return_counts=True)
                    for val, count in zip(unique, counts):
                        print(f"Value {val} fills {count / len(segment):.2%} of the segment")
                        
                    
            if N is None:
                y[row['start_frame']:int(row['end_frame'])] = row['code']
            else:
                try:
                    y[int(np.floor(row['start_percentile']*N)):int(np.ceil(row['end_percentile'] * N))] = row['code']
                except:
                    print(row['code'], N, int(np.floor(row['start_percentile']*N)), int(np.ceil(row['end_percentile'] * N)))
                    display(row.to_frame())
                    return np.nan
        elif row['type'] == 'point':
            margin_frame = int(2 * row.fps)
            if verbose:
                pass#print('margin_frame', margin_frame)
            
            if N is None:
                y[row['start_frame'] - margin_frame , row['start_frame'] + margin_frame ] = row['code']
            else:
                
                y[int(np.floor( ( (row['start_frame'] - margin_frame) / row['n_frames_max']   * N))) :int(np.ceil( (row['start_frame'] + margin_frame) / row['n_frames_max'] * N))] = row['code']
            
            
    
    if verbose:
        fi(20, 3)
        plt.scatter(np.arange(len(y)), y)
        participant_id = adf.participant_id.iloc[0]
        modality = adf.modality.iloc[0]
        plt.title(f'Participant {participant_id} {modality} - {mapping_categorie_track[category]}')
    
    return y
        
def add_ground_truth_labels(df, verbose=False):

    green('Adding ground truth labels to the dataframe: embedding_labels_x and segments_labels_x (maj voting without "non-annotation frames) :-)')
    _, _, _, mapping_categorie_track, _  = get_annotation_constants()
    
    assert 'N' in df.columns, "N not found in the dataframe"
    
    for category in mapping_categorie_track.keys():
        if category in ['E', 'F']:
            continue
        df[f'embedding_labels_{category}'] = df.apply(lambda row: create_label_array_from_dataframe(row.annotation.df, category=category, N=row.N, verbose=verbose) if row.annotation.has_annotation else np.nan, axis=1)
        df[f'segments_labels_{category}'] = df.apply(lambda row: predict_segments_from_embed_labels(row, temporal_segmentation_col='cpts', embedding_labels_col=f'embedding_labels_{category}', combine_func_name="majority_voting") if row.annotation.has_annotation else np.nan, axis=1)

    return df

def get_annotation_constants():

    mapping = pd.read_csv(os.path.join(get_data_root(), 'dataframes/tableau_annotation_cuisine_Smartflat.csv'), sep=";")

    mapping_dict = {}
    j=0
    order_categories = []
    for i, row in mapping.iterrows():
        
        if type(row["Encodage"])!= str and np.isnan(row["Encodage"]):
            continue
        else:
            j+=1
            #mapping_dict[row["Encodage"]] = {'semantic': row["Encodage"][0] + ': ' +row["Categorie"],
            #                                'code':  j}
            
            order_categories.append(row["Encodage"][0] + ': ' +row["Categorie"])
            
    mapping_categorie_track = {'A': 'Etapes de la tache',
                               'B': "Touche d'un object/ingredient",
                               'C': "Prise d'information visuelle",                        
                               'D': 'Actions sans buts',                  
                               'E': 'Evenements',
                               'F': 'Langages',                          
                               'G': 'Erreurs Comportementales',      
                               'H': 'Erreurs Neuropsychologiques',
                               'I': 'Autre',
                               'J': 'Evaluation Totale'
                              }
            
            
    cuisine_mapping_dict = {
                    'A0': {'semantic': 'A: Deplacement dans la salle', 'code': 1, 'color': 0}, 
                    'A1': {'semantic': 'A: Présentation du test - consignes', 'code': 2, 'color': 1},
                    
                    # Cuisine 
                    'A2': {'semantic': 'A: Lecture initiale de la recette', 'code': 3, 'color': 2},
                    'A3': {'semantic': 'A: Faire fondre le beurre-chocolat', 'code': 4, 'color': 3},
                    'A4': {'semantic': "A: Mélange farine-jaune d'oeuf-sucre", 'code': 5, 'color': 4},
                    'A5': {'semantic': 'A: Mélange beurre-chocolat', 'color': 5, 'code': 6},
                    'A6': {'semantic': 'A: Blancs en neige', 'code': 7, 'color': 6},
                    'A7': {'semantic': "A: Mélange adf blancs d'oeufs à la pâte", 'code': 8, 'color': 7},
                    'A8': {'semantic': 'A: Mise dans le moule', 'code': 9, 'color': 8},
                    'A9': {'semantic': 'A: Cuisson', 'code': 10, 'color': 9},
                    'A10': {'semantic': 'A: Melange beure-chocolat et farine-oeuf-sucre', 'code': 47, 'color': 10},
                    
                    'A11': {'semantic': 'A: Sonnerie du four', 'code': 31, 'color':11}, # Cuisine
                    'E1': {'semantic': 'A: Sonnerie du four', 'code': 31, 'color':11}, # Cuisine

                    
                    # Cuisine 
                    'B1': {'semantic': 'B: Beurre', 'code': 11, 'color': 0},
                    'B2': {'semantic': 'B: Chocolat', 'code': 12, 'color': 1},
                    'B3': {'semantic': 'B: Farine', 'code': 13, 'color': 2},
                    'B4': {'semantic': 'B: Sucre', 'code': 14, 'color': 3},
                    'B5': {'semantic': 'B: Oeuf', 'code': 15, 'color': 4},
                    'B6': {'semantic': 'B: Saladier', 'code': 16, 'color': 5},
                    'B7': {'semantic': 'B: Ustensile', 'code': 17, 'color': 6},
                    'B8': {'semantic': 'B: Balance', 'code': 18, 'color': 7},
                    'B9': {'semantic': 'B: Battteur', 'code': 19, 'color': 8},
                    'B10': {'semantic': 'B: Placard', 'code': 20, 'color': 9},
                    'B11': {'semantic': 'B: Frigo', 'code': 21, 'color': 10},
                    'B12': {'semantic': 'B: Evier', 'code': 22, 'color': 11},
                    'B13': {'semantic': 'B: Moule', 'code': 23, 'color': 12},
                    'B14': {'semantic': 'B: Four', 'code': 24, 'color': 13},
                    'B15': {'semantic': 'B: Manipulation', 'code': 0, 'color': 14}, # For Boris codes
                    
                    
                    'C1': {'semantic': 'C: Lecture de la recette', 'code': 25, 'color': 0}, # Cuisine
                    'C2': {'semantic': 'C: Fixation', 'code': 26, 'color': 1},
                    'C3': {'semantic': 'C: Exploration visuelle', 'code': 27, 'color': 2},
                    'C4': {'semantic': 'C: Regard examinateur', 'code': 28, 'color': 3},
                    
                    'D1': {'semantic': 'D: Manipule sans utiliser', 'code': 29, 'color': 0},
                    'D2': {'semantic': 'D: Deplacement', 'code': 30, 'color': 1},
                    'D3': {'semantic': 'D: Betise auto-corrige', 'code': 45, 'color':1}, # Question - Ask Flavie if should be H or G? 
                    'D4': {'semantic': 'D: Intervention examinateur', 'code': 49, 'color': 3}, # For Boris codes

                    
                    
                    
                    
                    'F1': {'semantic': 'F: Aux autres', 'code': 32, 'color': 1},
                    'F2': {'semantic': 'F: A soi-meme', 'code': 33, 'color': 2},
                    
                    'G1': {'semantic': 'G: Commentaires/Questions', 'code': 34, 'color': 0},
                    'G2': {'semantic': 'G: Addition', 'code': 35, 'color': 1},
                    'G3': {'semantic': 'G: Estimation', 'code': 36, 'color': 2},
                    'G4': {'semantic': 'G: Omission', 'code': 37, 'color': 3},
                    'G5': {'semantic': 'G: Substitution/Inversion', 'code': 38, 'color': 4},
                    
                    'H1': {'semantic': 'H: Adhérence environnementale', 'code': 39, 'color': 0},
                    'H2': {'semantic': 'H: Négligence du contexte', 'code': 40, 'color': 1},
                    'H3': {'semantic': 'H: Aide/Help/dependance', 'code': 41, 'color': 2},
                    'H4': {'semantic': 'H: Errance et perplexité, action sans but', 'code': 42, 'color': 3},
                    'H5': {'semantic': 'H: Trouble du comportement', 'code': 43, 'color': 4},
                    'H6': {'semantic': 'H: Vérification/contrôle', 'code': 44, 'color': 5},

                    'J0': {'semantic': 'J: Evaluation totale', 'code': 48, 'color': 0},
                    
                    'I2': {'semantic': 'I: Unknown action', 'code': 46, 'color':2}, # For Boris codes
                    }
    
    
                            
                            
                            
    lego_mapping_dict = {
                    
                    'A0': {'semantic': 'A: Deplacement dans la salle', 'code': 1, 'color': 0}, 
                    'A1': {'semantic': 'A: Présentation du test - consignes', 'code': 2, 'color': 1},
                    #'A2': {'semantic': 'A: Lecture initiale de la consigne', 'code': 3, 'color': 2}, # present at all ? 
                    'A3': {'semantic': 'A: Pylône 1', 'code': 4, 'color': 3},
                    'A4': {'semantic': "A: Arc", 'code': 5, 'color': 4},
                    'A5': {'semantic': 'A: Pylône 2', 'color': 5, 'code': 5},
                    'A6': {'semantic': 'A: Attique', 'code': 7, 'color': 6},
                    'A7': {'semantic': "A: Finitions", 'code': 8, 'color': 7},
                    'A8': {'semantic': "A: Points Saillants", 'code': 9, 'color': 8},

                    'B7': {'semantic': 'B: Manipulation', 'code': 0, 'color': 0}, # TODO: check if those having the B7 code aren't cooking ? 
                    'B15': {'semantic': 'B: Manipulation', 'code': 0, 'color': 0}, # 

                    'C1': {'semantic': 'C: Lecture de la consigne', 'code': 25, 'color': 0}, # Cuisine
                    'C2': {'semantic': 'C: Fixation', 'code': 26, 'color': 1},
                    'C3': {'semantic': 'C: Exploration visuelle', 'code': 27, 'color': 2},
                    'C4': {'semantic': 'C: Regard examinateur', 'code': 28, 'color': 3},
                    
                    'D1': {'semantic': 'D: Manipule sans utiliser', 'code': 29, 'color': 0},
                    'D2': {'semantic': 'D: Deplacement', 'code': 30, 'color': 1},
                                        
                    'F1': {'semantic': 'F: Aux autres', 'code': 32, 'color': 0},
                    'F2': {'semantic': 'F: A soi-meme', 'code': 33, 'color': 1},
                    
                    'G1': {'semantic': 'G: Commentaires/Questions', 'code': 34, 'color': 0},
                    'G2': {'semantic': 'G: Addition', 'code': 35, 'color': 1},
                    'G3': {'semantic': 'G: Estimation', 'code': 36, 'color': 2},
                    'G4': {'semantic': 'G: Omission', 'code': 37, 'color': 3},
                    'G5': {'semantic': 'G: Substitution/Inversion', 'code': 38, 'color': 4},
                    
                    'H1': {'semantic': 'H: Adhérence environnementale', 'code': 39, 'color': 0},
                    'H2': {'semantic': 'H: Négligence du contexte', 'code': 40, 'color': 1},
                    'H3': {'semantic': 'H: Aide/Help/dependance', 'code': 41, 'color': 2},
                    'H4': {'semantic': 'H: Errance et perplexité, action sans but', 'code': 42, 'color': 3},
                    'H5': {'semantic': 'H: Trouble du comportement', 'code': 43, 'color': 4},
                    'H6': {'semantic': 'H: Vérification/contrôle', 'code': 44, 'color': 5},
                    
                    'J0': {'semantic': 'J: Evaluation totale', 'code': 48, 'color': 0},
                    'D3': {'semantic': 'D: Betise auto-corrige', 'code': 45, 'color':1}, # Question - Ask Flavie if should G (cf arbitrage changementd de page, mild errors) 
                    'I2': {'semantic': 'I: Unknown action', 'code': 46, 'color':2}, # For Boris codes e.g. digression/autre actions non prevu. (ot addition)
                    'D4': {'semantic': 'D: Intervention examinateur', 'code': 47, 'color': 3}, # For Boris codes

                    }

    
    
    # Create mapping dictionary between categories of Boris and vidat annotation 
    

    mapping_boris_vidat = {'Te. - Test Gateau en cours': 'J0', # Question comparison: here this is supposed to span all the assessment by the participant (from initial reading to puting the cake in the oven)
                           'Te. - Test': 'J0', 
                            'T. - test Lego': 'J0', 
                            'T. - Test': 'J0',
                            
                            'O. - Presentation': 'A1',
                            'O. - Presentation de la tache': 'A1', 
                            
                            'Te. - P1 : Faire fondre chocolat et beurre': 'A3',
                            "Te. - P2 : Melanger farine, sucre et jaunes d'oeufs": 'A4',
                            "Te. - P3 : Melanger farine, sucre et jaunes d'oeufs": 'A4',
                            'Te. - P2 : Melanger chocolat et beurre': 'A5',
                            'Te. - P3 : Melanger chocolat et beurre': 'A5',
                            "Te. - P5 : Battre les blancs d'oeufs en neige": 'A6',
                            'Te. - P6 : Incorporation des blancs au mélange précédent': 'A7',
                            'Te. - P6 : Incorporation des blancs au melange precedent': 'A7',
                            'Te. - P7 : Mise dans le moule de la préparation': 'A8',
                            
                            'Te. - P8 : Mise au four du moule et programmation': 'A9',
                            'Te. - P4 : Melanger les deux mélanges précédents': 'A10', # Boris annotation add this step
                            'Te. - P4 : Melanger les deux melanges precedents': 'A10',
                           

                            'T. - P1 : Pylône 1': 'A3', 
                            'T. - P1 : Pylône 1 gauche': 'A3',
                            'T. - P2 : Arc': 'A4',
                            'T. - P2 : Arc central': 'A4',
                            'T. - P3 : Pylône 2': 'A5',
                            'T. - P3 : Pylône 2 droit': 'A5',
                            'T. - P4 : Attique': 'A6',
                            'T. - P5 : Finitions': 'A7',
                            
                            'T. - points Saillants': 'A8', # 
                            'T. - Points Saillants': 'A8', # 
                            
                            
                            "Ta. - Manipulation d'ustensiles": 'B7',
                            
                            'T. - Manipulation de piece': 'B15', 
                            'T. - Manipulation de piece lego': 'B15',
                            'T. - manipulation de piece lego': 'B15',
                            "Ta. - preparation et manipulation d'ingredients": 'B15', # Question check that it is B15 ()

                            'Ta. - Regard recette': 'C1',
                            'T. - regard Notice': 'C1',
                            'Ta. - Recette': 'C1',
                            'T. - Notice': 'C1',
                            
                            'O. - déplacement dans la piece': 'D2',
                             
                            'Te. - sonnerie four': 'A11',
                            'sonnerie four': 'A11',
                            'O. - Intervention examinateur': 'D4',
                            'O. - Intervention': 'D4', 

                             
                            'E.D. - Commentaires et Questions': 'G1',
                            'E.D. - Commentaires et questions': 'G1',
                            'E.D. - Addition': 'G2',
                            'E.D. - Estimation': 'G3',
                            'E.D. - Omission': 'G4',
                            'E.D. - Substitution et inversion': 'G5',

                            'E.N. - aDhérence Environementale': 'H1',
                            'E.N. - Adhérence Environementale': 'H1',
                            'E.N. - Adhérence': 'H1',
                            'E.N. - aDhérence_persévère': 'H1',
                            'E.N. - Négligence du Contexte': 'H2',
                            'E.N. - Contexte': 'H2',
                            'E.N. - Contexte_consignes_environnement': 'H2',
                            "E.N. - Demande d'aide/Dependency": 'H3',
                            "E.N. - Demande d'aide": 'H3',
                            "E.N. - Demande d'aide Help":'H3', 
                            'category: E.N. - Errance': 'H4',
                            'E.N. - eRrance': 'H4',
                            'E.N. - Errance': 'H4',   
                            'E.N. - eRrance_comportement inutile sans but': 'H4',
                            'E.N. - Trouble du comportement': 'H5',
                            'E.N. - Controle des erreurs/Vérification': 'H6',
                            'E.N. - Controle des erreurs/vérification': 'H6',
                            'E.N. - Verification_controle_stratégie': 'H6',
                            'E.N. - Verification': 'H6',
                            
                            'betise_erreur autocorrigee': 'D3', # New category
                            'E.D. -Autocorrection-Betise': 'D3', # New category
                            'Autre action Unknown': 'I2', # New category
                            'Autre action': 'I2', # New category 
                          }

    return order_categories, cuisine_mapping_dict, lego_mapping_dict, mapping_categorie_track, mapping_boris_vidat

