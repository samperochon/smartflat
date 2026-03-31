import json
import os
import sys
from copy import deepcopy
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

# add tools path and import our own tools


from const import *
from utils import (
    CostCustom,
    create_frames_label,
    create_xticklabels,
    decompose,
    extend_labels,
    fi,
    pairwise,
)

# TODO: TWFINCH dependency removed — install as proper package if needed
# from python.twfinch import FINCH

overlaps = [.1, .25, .5]

class Annotation(object):
    """
        Class handling all the annotations, whether there are grund truth or predicted.
        Time reference is always the one of the video (no sampling). 
        The start frames and stop frames always contains the video first and last frames.

    """

    def __init__(self,
                names = ['Ground Truth', 'Prediction (ours)'],
                overlap_list=[.1, .25, .5],
                boris=True): #'boris' or 'json'
        """ 
            We assume here that if the cpt are provided, they contain a [0] and [n_frame] values.
            If we provide start and end frames, we add [0] and [n_frames]
        """

       
        self.gt_available = False
        self.prediction_is_available = True 

        self.pred_frames_label = None
        self.gt_label = None
        
        self.names = names
        self.overlap_list = overlap_list
        
        # Segment-wise metrics
        self.f1_10 = None
        self.f1_25 = None
        self.f1_50 = None
        self.map = None
        self.map_10 = None 
        self.map_20 = None 
        self.map_30 = None 
        self.map_40 = None  
        self.map_50 = None 
        
        
        # Frame-wise metrics
        self.accuracy = None
        self.f1_macro = None
        self.iou = None


    def add(self, pred_frames_label):
    
        if not isinstance(pred_frames_label, list):
            pred_frames_label = list(pred_frames_label)
        
        if self.gt_label is not None and len(pred_frames_label) >= len(self.gt_label):
            self.pred_frames_label = (np.array(pred_frames_label)[:len(self.gt_label)]).astype(int)
        elif self.gt_label is not None:
            self.pred_frames_label = np.concatenate([pred_frames_label + [pred_frames_label[-1]] * (len(self.gt_label)-len(pred_frames_label))], axis=0).astype(int)

        else:
            self.pred_frames_label = np.array(pred_frames_label).astype(int)
            
        # Set the prediction to span successive integers (required for the Hungarian matching)
        mapping_original_ordinal_pred = {original: ordinal for ordinal, original  in enumerate(np.unique(self.pred_frames_label))}
        self.pred_frames_label = np.array([mapping_original_ordinal_pred[l] for l in self.pred_frames_label.astype(int)])

        self.prediction_is_available = True 

        return   

    def compute_metrics(self, fps):

        if self.gt_label is None:
            print("No ground truth added.")
            return 

        if self.pred_frames_label is None:
            print("No prediction added.")
            return 


        # Find best assignment through Hungarian Method
        cost_matrix = estimate_cost_matrix(self.gt_label, self.pred_frames_label)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # decode the predicted labels
        self.pred_frames_label = col_ind[self.pred_frames_label]

        #-------------------- Compute Frame-wise metrics ------------------ #
        
        # Calculate the metrics (External libraries)
        self.accuracy = metrics.accuracy_score(self.gt_label, self.pred_frames_label)

        # F1-Score
        self.f1_macro = metrics.f1_score(self.gt_label, self.pred_frames_label, average='macro')  

        # Jaccard index or iou
        self.iou = np.sum(metrics.jaccard_score(self.gt_label, self.pred_frames_label, average=None)) / len(np.unique(self.gt_label))

        # Compute edit distance 
        self.edit = levenstein(self.pred_frames_label, self.gt_label)

        #-------------------- Compute Segment-wise metrics ------------------ #
        self.f1_10, self.f1_25, self.f1_50 = self.compute_segmental_f1(self.gt_label, self.pred_frames_label, overlaps=[0.1, 0.25, 0.5])

        # Map over 0.1-0.5 overlaps
        self.map, self.map_10, self.map_20, self.map_30, self.map_40, self.map_50 = compute_map(self.pred_frames_label, self.gt_label, fps=fps, overlaps=[0.1, 0.2, 0.3, 0.4, 0.5])

        return

    def compute_segmental_f1(self, frame_gt, frame_prediction, overlaps):

        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        # Add tp, fp and fn for f1 computation
        for s in range(len(overlaps)):
            tp1, fp1, fn1 = f_score(frame_prediction, frame_gt, overlaps[s])
            print("Overlap: {} TP: {} FP: {} FN: {}".format(overlaps[s], tp1, fp1, fn1))
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        f1s = np.array([0, 0 ,0], dtype=float)
        for s in range(len(overlaps)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            f1s[s] = f1

        return f1s[0], f1s[1], f1s[2]

    def plot(self, names=None, cpts=[], title='Ours', plot_gt=True):

        """
        if names is None:
            names = self.names
        
        fig, ax = plt.subplots(figsize=(30,5))

        (ymin1, ymax1, ymin_2, ymax2) = (0.55, .95, .05, .47) if (self.is_available and self.prediction_is_available) else (0.05, .95, None, None)

        title=""


        if self.is_available:
            p, pred_start_frame, pred_stop_frame = get_labels_start_end_time(self.pred_frames_label, bg_class=[0])
            y, gt_start_frame, gt_stop_frame = get_labels_start_end_time(self.gt_label, bg_class=[0])


            gt_tuple = [(start_frame, stop_frame-start_frame) for start_frame, stop_frame in zip(gt_start_frame, gt_stop_frame)]
            ax.broken_barh(gt_tuple, (20, 9), facecolors='tab:green', alpha=.5, label='Ground Truth')

            for start_frame, stop_frame in zip(gt_start_frame, gt_stop_frame):

                ax.axvline(x=start_frame, ymin=ymin1, ymax=ymax1, color='k',linestyle='-.')
                ax.axvline(x=stop_frame, ymin=ymin1, ymax=ymax1, color='k',linestyle='-.')
            title += 'True Segmentation (green)\n'

        if self.prediction_is_available:
            
            pred_tuple = [(start_frame, stop_frame-start_frame) for start_frame, stop_frame in zip(pred_start_frame, pred_stop_frame)]
            ax.broken_barh(pred_tuple, (10, 9),facecolors=('tab:red'), alpha=.5, label='Prediction')

            for start_frame, stop_frame in zip(pred_start_frame, pred_stop_frame):
                ax.axvline(x=start_frame, ymin=ymin_2, ymax=ymax2, color='k',linestyle='-.')
                ax.axvline(x=stop_frame, ymin=ymin_2, ymax=ymax2, color='k',linestyle='-.')
            
            title += 'Prediction - {} (red)\n'.format(names[1])

        for cpt in cpts:
            ax.axvline(x=cpt, ymin=ymin1, ymax=ymax1, color='gray',linestyle='-')

        ax.set_xlabel('Time [Frame]')#;ax.set_yticks([])
        ax.set_title(title, weight='bold', fontsize=18);ax.set_xlabel('Time [Frame]', weight='bold', fontsize=18);ax.legend()
        plt.show()

        """

        if self.gt_label is not None and plot_gt:
            plt.figure(figsize=(25, 5))
            plt.imshow(np.repeat(self.gt_label[np.newaxis, :], 1000, axis=0), cmap='tab20');plt.title('Ground Truth');plt.show()
        if self.pred_frames_label is not None:
            plt.figure(figsize=(25, 5))
            plt.imshow(np.repeat(self.pred_frames_label[np.newaxis, :], 1000, axis=0), cmap='tab20');plt.title(title)

        plt.show() 
        return 

    def add_epic_kitchen(self, video_name, n_frames, granularity='action', filepath="/home/perochon/EPIC_100_train.csv"):

        annotation_data = pd.read_csv(filepath)

        subject_data = annotation_data.query(" `video_id` == @video_name")
        subject_data.sort_values(by='start_timestamp', inplace=True)

        self.df = subject_data

        if granularity=='action':

            mapping_actions = {act:i+1 for i, act in enumerate(np.unique(subject_data['narration']))}
            subject_data['action_class'] = [mapping_actions[act] for act in subject_data['narration'].tolist()]
            gt_label = np.zeros(n_frames)
            for i, row in subject_data.iterrows(): 
                gt_label[row['start_frame']:row['stop_frame']] = row['action_class']
            self.gt_label = gt_label.astype(int)

        if granularity=='verb':

            mapping_actions = {act:i+1 for i, act in enumerate(np.unique(subject_data['verb_class']))}
            subject_data['verb_class'] = [mapping_actions[act] for act in subject_data['verb_class'].tolist()]
            gt_label_verb = np.zeros(n_frames)
            for i, row in subject_data.iterrows():
                gt_label_verb[row['start_frame']:row['stop_frame']] = row['verb_class']
            self.gt_label = gt_label_verb.astype(int)

        if granularity=='noun':

            mapping_actions = {act:i+1 for i, act in enumerate(np.unique(subject_data['noun_class']))}
            subject_data['noun_class'] = [mapping_actions[act] for act in subject_data['noun_class'].tolist()]
            gt_label_noun = np.zeros(n_frames)
            for i, row in subject_data.iterrows():
                gt_label_noun[row['start_frame']:row['stop_frame']] = row['noun_class']
            self.gt_label = gt_label_noun.astype(int)

        self.gt_start_frame = subject_data['start_frame'].tolist()
        self.gt_stop_frame = subject_data['stop_frame'].tolist()
        self.is_available = True

        return 

    def add_benchmark(self, dataset_name, video_name, n_frames=None, granularity='action'):

        self.dataset_name = dataset_name
        self.video_name = video_name

        if dataset_name in ['Breakfast']:

            # setup paths
            datasets_path = '/home/perochon/temporal_segmentation/data/Action_Segmentation_Datasets/'
            path_ds = os.path.join(datasets_path, dataset_name)
            path_gt = os.path.join(path_ds, 'groundTruth/')
            path_mapping = os.path.join(path_ds, 'mapping', 'mapping.txt')

            # %% Load all needed files: Descriptor, GT & Mapping
            if os.path.exists(path_mapping):
                # Create the Mapping dict
                mapping_dict = get_mapping(path_mapping)


            # Load the gt_label, map them to the corresponding ID    
            gt_label_path = os.path.join(path_gt, video_name)
            if not os.path.exists(gt_label_path):
                gt_label_path = os.path.join(path_gt, video_name + '.txt')

                if dataset_name == 'Breakfast':
                    gt_label_path, self.video_name = check_gt_path(gt_label_path, self.video_name)

            self.gt_label, _ = read_gt_label(gt_label_path, mapping_dict=mapping_dict)



        if dataset_name == '50Salads':

            if granularity == 'low':
                print("Error for this level...")
                file = '/home/perochon/temporal_segmentation/data/Action_Segmentation_Datasets/50Salads/annotation_low.csv'
                df = pd.read_csv(file)
                self.df = df[df['video_name']==self.video_name[4:]]

                gt_label = np.zeros(n_frames)

                
                mapping_actions = {act:i+1 for i, act in enumerate(np.unique(self.df['low_label']))}
                self.df['low_label'] = [mapping_actions[act] for act in self.df['low_label'].tolist()]
                gt_label = np.zeros(n_frames)

                for i, row in self.df.sort_values(by='Start Frame').iterrows():

                    gt_label[row['Start Frame']:row['Stop Frame']] = row['low_label']

                self.gt_label = np.array(gt_label).astype(int)
                return 

            if granularity == 'mid':
                maping_file_path = '/home/perochon/temporal_segmentation/data/50Salads/ann-ts/mapping.txt'
                df = pd.read_csv(maping_file_path, sep=" ", header=None, names=['index', 'action_name'])
                mapping_dict = dict(zip(df['action_name'], df['index']))

            elif granularity == 'high':
                maping_file_path = '/home/perochon/temporal_segmentation/data/50Salads/ann-ts/mappinghigh.txt'
                df = pd.read_csv(maping_file_path, sep=" ", header=None, names=['index', 'action_name'])
                mapping_dict = dict(zip(df['action_name'], df['index']))

            elif granularity == 'eval':
                maping_file_path = '/home/perochon/temporal_segmentation/data/50Salads/ann-ts/mappingeval.txt'
                df = pd.read_csv(maping_file_path, sep=" ", header=None, names=['index', 'action_name'])
                mapping_dict = dict(zip(df['action_name'], df['index']))


            # setup paths
            datasets_path = '/home/perochon/temporal_segmentation/data/Action_Segmentation_Datasets/'
            path_ds = os.path.join(datasets_path, dataset_name)
            path_gt = os.path.join(path_ds, 'groundTruth/')

            # Load the gt_label, map them to the corresponding ID    
            gt_label_path = os.path.join(path_gt, video_name)
            if not os.path.exists(gt_label_path):
                gt_label_path = os.path.join(path_gt, video_name + '.txt')

            self.gt_label, _ = read_gt_label(gt_label_path, mapping_dict=mapping_dict)





        if dataset_name == 'GTEA':

            annotation_path = '/home/perochon/temporal_segmentation/data/data_features/{}_annotation.csv'.format(dataset_name)
            df = pd.read_csv(annotation_path)
            self.df = df[df['video_name']==self.video_name]

            gt_label = np.zeros(n_frames)
        
            if granularity == 'action':
    
                
                mapping_actions = {act:i+1 for i, act in enumerate(np.unique(self.df['gt_label']))}
                self.df['gt_label'] = [mapping_actions[act] for act in self.df['gt_label'].tolist()]
                gt_label = np.zeros(n_frames)
                
                for i, row in self.df.sort_values(by='Start Frame').iterrows():

                    gt_label[row['Start Frame']:row['Stop Frame']] = row['gt_label']

            elif granularity == 'verb':
                
                mapping_actions = {act:i+1 for i, act in enumerate(np.unique(self.df['verb_label']))}
                self.df['verb_label'] = [mapping_actions[act] for act in self.df['verb_label'].tolist()]
                gt_label = np.zeros(n_frames)
                
                for i, row in self.df.sort_values(by='Start Frame').iterrows():

                    gt_label[row['Start Frame']:row['Stop Frame']] = row['verb_label']

            elif granularity == 'noun':
                
                mapping_actions = {act:i+1 for i, act in enumerate(np.unique(self.df['noun_label']))}
                self.df['noun_label'] = [mapping_actions[act] for act in self.df['noun_label'].tolist()]
                gt_label = np.zeros(n_frames)
                
                for i, row in self.df.sort_values(by='Start Frame').iterrows():

                    gt_label[row['Start Frame']:row['Stop Frame']] = row['noun_label']

            self.gt_label = np.array(gt_label).astype(int)
            self.mapping_original_ordinal_gt = {original: ordinal for ordinal, original  in enumerate(np.unique(self.gt_label))}

        annotation_path = '/home/perochon/temporal_segmentation/data/data_features/{}_annotation.csv'.format(dataset_name)
        df = pd.read_csv(annotation_path)
        self.df = df[df['video_name']==self.video_name]

        self.is_available=True
        return 

    def cpt_from_start_stop(self, start_frame, stop_frame):


        cpt = np.sort(np.unique(np.concatenate([start_frame, stop_frame])))

        # In case the cpt are consecutive, we only want one rupture
        return cpt[np.argwhere(np.ediff1d(cpt) > 2)].squeeze()

def estimate_cost_matrix(gt_label, cluster_labels):
    # Make sure the lengths of the inputs match:
    if len(gt_label) != len(cluster_labels):
        print('The dimensions of the gt_labls and the pred_labels do not match')
        return -1
    L_gt = np.unique(gt_label)
    L_pred = np.unique(cluster_labels)
    nClass_pred = len(L_pred)
    dim_1 = max(nClass_pred, np.max(L_gt) + 1)
    profit_mat = np.zeros((nClass_pred, dim_1))
    
    for i, frame_pred in enumerate(L_pred):
        idx = np.argwhere(cluster_labels == frame_pred)
        gt_selected = np.array(gt_label)[idx]
        for j, frame_gt in enumerate(L_gt):
            profit_mat[i][j] = np.count_nonzero(gt_selected == frame_gt)
            
    return -profit_mat


def levenstein(frame_prediction, frame_gt, norm=True):

    p, _, _ = get_labels_start_end_time(frame_prediction, bg_class=[0])
    y, _, _ = get_labels_start_end_time(frame_gt, bg_class=[0])

    n_clusters = len(np.unique(y))
    
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                            D[i, j-1] + 1,
                            D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score    


def get_mapping(maping_file_path):
    df = pd.read_csv(maping_file_path, sep=" ", header=None, names=['index', 'action_name'])
    mapping_dict = dict(zip(df['action_name'], df['index']))
    return mapping_dict


def check_gt_path(gt_label_path, video_name):
    
    if os.path.basename(gt_label_path).split('.')[-2][-3:-1] == 'ch':
        basename = os.path.basename(gt_label_path).split('.')[-2][:-4]
        paths = glob(os.path.join(os.path.join(os.path.dirname(gt_label_path), os.path.basename(gt_label_path).split('_')[0] + '_ster*' + os.path.basename(gt_label_path).split('_')[-2] +'*')))
        if len(paths) == 1:
            gt_label_path  = paths[0]
            video_name = os.path.basename(gt_label_path)
                        
    return gt_label_path, video_name


def read_gt_label(gt_label_path, mapping_dict=None):
    
    df_gt = pd.read_csv(gt_label_path, sep=" ", header=None)
    
    gt = df_gt[0].tolist()
    if mapping_dict is not None:
        gt_label = [mapping_dict[i] for i in gt]
        gt_label = np.array(gt_label)
        n_labels = len(mapping_dict)
    else:
        _, gt_label = np.unique(gt, return_inverse=True)
        n_labels = len(np.unique(gt_label))

    # make sure gt label do not contain -ve entries
    gt_min = np.min(gt_label)
    if gt_min < 0:
        gt_label = gt_label - gt_min

    return gt_label, n_labels


def get_labels_start_end_time(frame_wise_labels, bg_class=[0]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends




def get_predictions_with_label(prediction_by_label, cidx):
    """Get all predicitons of the given label. Return empty DataFrame if there
    is no predcitions with the given label. 
    """
    try:
        return prediction_by_label.get_group(cidx).reset_index(drop=True)
    except:
        #print('Warning: No predictions of label \'%s\' were provdied.' % cidx)
        return pd.DataFrame()
    
def compute_map(pred, gt, overlaps, fps):

    labels_pred, p_start, p_end = get_labels_start_end_time(pred, bg_class=[0])
    labels_gt, p_start_gt, p_end_gt = get_labels_start_end_time(gt, bg_class=[0])

    prediction = pd.DataFrame({'video-id':['_' for _ in range(len(labels_pred))],
                               'label' : [label for label in labels_pred],
                               't-start' : [t/fps for t in p_start],
                               't-end' : [t/fps for t in p_end],
                               'score' : [1 for _ in range(len(labels_pred))]})

    ground_truth = pd.DataFrame({'video-id':['_' for _ in range(len(labels_gt))],
                                 'label' : [label for label in labels_gt],
                                   't-start' : [t/fps for t in p_start_gt],
                                   't-end' : [t/fps for t in p_end_gt],
                                   'score' : [1 for _ in range(len(labels_gt))]})

    
    activity_index = np.unique(labels_gt)
    ap = np.zeros((len(overlaps), len(activity_index)))

    # Adaptation to query faster
    ground_truth_by_label = ground_truth.groupby('label')
    prediction_by_label = prediction.groupby('label')

    for i, cidx in enumerate(activity_index):

        gt = ground_truth_by_label.get_group(cidx).reset_index(drop=True)

        pred = get_predictions_with_label(prediction_by_label, cidx)

        ap[:,i] = compute_average_precision_detection(gt, pred, tiou_thresholds=overlaps)
    mAP = ap.mean(axis=1)
    average_mAP = mAP.mean()
    
    return average_mAP, mAP[0], mAP[1], mAP[2], mAP[3], mAP[4]


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap



def f_score(recognized, ground_truth, overlap, bg_class=[0]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)




def _add_vidat_old(self, filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    fps = data['annotation']['video']['fps']
    self.df = pd.DataFrame(columns=['Scene Number', 'Start Frame', 'Start Time (seconds)', 'End Frame', 'End Time (seconds)', 'Length (frames)', 'Length (seconds)'])

    for i, annot_dict in enumerate(data['annotation']['actionAnnotationList']):

        self.df = self.df.append(pd.DataFrame({'Scene Number': i, 
                                                'Start Frame': np.round(annot_dict['start']*fps).astype(int), 
                                                'Start Time (seconds)': annot_dict['start'], 
                                                'End Frame': np.round(annot_dict['end']*fps).astype(int), 
                                                'End Time (seconds)': annot_dict['end'], 
                                                'Length (frames)': np.round((annot_dict['end'] - annot_dict['start'])).astype(int), 
                                                'Length (seconds)': (annot_dict['end'] - annot_dict['start']), 
                                                }, index=[i]))

    self.gt_start_frame = self.df['Start Frame'].to_numpy()
    self.gt_stop_frame = self.df['End Frame'].to_numpy()
    self.is_available=True

    return 



def parse_vidat(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    mapping_action_vidat = {}
    for dict_action in data['config']['actionLabelData']:

        mapping_action_vidat[dict_action['id']] = {'label': dict_action['name'],
                                                   'color': dict_action['color']
                                                  }
    mapping_categorie_track = {'A': 'Etapes de la recette',
                               'B': "Touche d'un object/ingredient",
                               'C': "Prise d'information visuelle",                        
                               'D': 'Actions sans buts',                  
                               'E': 'Evenements',
                               'F': 'Langages',                          
                               'G': 'Erreurs Comportementales',      
                               'H': 'Erreurs Neuropsychologiques'
                              }

    fps = data['annotation']['video']['fps']
    n_frames = data['annotation']['video']['frames']

    df = pd.DataFrame(columns=['Scene Number', 'Start Frame', 'Start Time (seconds)', 'End Frame', 'End Time (seconds)', 'Length (frames)', 'Length (seconds)'])

    for i, annot_dict in enumerate(data['annotation']['actionAnnotationList']):

        df = df.append(pd.DataFrame({'Scene Number': i, 
                                        'Start Frame': np.round(annot_dict['start']*fps).astype(int), 
                                        'Start': annot_dict['start'], 
                                        'End Frame': np.round(annot_dict['end']*fps).astype(int), 
                                        'End': annot_dict['end'], 
                                        'Length (frames)': np.round((annot_dict['end'] - annot_dict['start'])).astype(int), 
                                        'Length': (annot_dict['end'] - annot_dict['start']), 
                                        'Color': annot_dict['color'], 
                                        'label': mapping_action_vidat[annot_dict['action']]['label'],
                                        'Semantic': mapping_dict[mapping_action_vidat[annot_dict['action']]['label']]['semantic'],
                                        'code': int(mapping_dict[mapping_action_vidat[annot_dict['action']]['label']]['code']),
                                        'Categorie':  mapping_action_vidat[annot_dict['action']]['label'][0],
                                        'Categorie Label': mapping_categorie_track[mapping_action_vidat[annot_dict['action']]['label'][0]],
                                        'Description': annot_dict['description'] if len(annot_dict['description']) > 0 else "None"
                                
                                        }, index=[i]))

    df['Length (s)'] = (df['End'] - df['Start']).round(2)
    df['Start'] = df['Start'].apply(lambda x: datetime.fromtimestamp(datetime(2022,1,1).timestamp()+x))
    df['End'] = df['End'].apply(lambda x: datetime.fromtimestamp(datetime(2022,1,1).timestamp()+x))

    df.drop_duplicates(subset=['Start Frame', 'End Frame'], inplace=True)
    df.sort_values(by='Start Frame', inplace=True)

    df.head()
    return df
