import os
import sys
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ruptures.metrics.sanity_check import sanity_check

# add tools path and import our own tools

from sklearn.metrics import f1_score, jaccard_score

overlaps = [.1, .25, .5]


class Metrics:

    def __init__(self, overlap_list=[.1, .25, .5], margin=10):

        self.margin = margin
        self.overlap_list = overlap_list
        self.scores = {'f1':{'overlap': [], 'tp': [], 'fp': [], 'fn': [], 'precision': [], 'recall': [], 'f1': []},
                       'accuracy':{'overlap': [], 'tp': [], 'fp': [], 'fn': [], 'precision': [], 'recall': [], 'f1': []}}

        self.gt_label = None
        self.gt_cpt = None
        self.gt_start_frame = None
        self.gt_stop_frame = None

        self.pred_label = None
        self.pred_cpt = None
        self.pred_start_frame = None
        self.pred_stop_frame = None


    def compute(self, method='f1'):
        
        if method=='f1':
            return self.compute_f1()
            
        elif method=='timestamps_f1':
            return self.compute_timestamps_f1()
            
        elif method=='accuracy':
            return self.compute_accuracy()

        else:
            raise NotImplementedError("The available mperformance metrics are: {} or {} or {}. :)".format('accuracy', 'f1', 'timestamps_f1'))
            
    def compute_f1(self, gt_label=None, pred_label=None, gt_start_frame=None, pred_start_frame=None, 
                    gt_stop_frame=None, pred_stop_frame=None, gt_cpt=None, pred_cpt=None, overlap_list=[]):
        
        """Calculate the precision/recall of an estimated segmentation compared
        with the true segmentation.=, taking into account the labels ofeach segments. 

        Args:
            gt_label (list): list of the label of each regimes (true partition).
            gt_start_frame (list): list of the start index of each regime (true
                partition).
            gt_stop_frame (list): list of the last index of each regime (computed
                partition).
            pred_label (list): list of the label of each regimes (computed partition).
            pred_start_frame (list): list of the start index of each regime (true
                partition).
            pred_stop_frame (list): list of the last index of each regime (computed
                partition).
            gt_cpt (list, optional): list of the change-points of the true partition. 
                Used to compute start and stop frames if needed.
            pred_cpt (list, optional): list of the change-points of the true partition. 
                Used to compute start and stop frames if needed.
            overlap (float, optional): allowed overlap when computing IoU.

        Returns:
            tuple: (precision, recall)
        """

        if not isinstance(overlap_list, list): 
            overlap_list = [overlap]
        self.overlap_list = overlap_list

        if gt_cpt is not None:
            gt_start, gt_end = self.decompose(gt_cpt)
            pred_start, pred_end = self.decompose(pred_cpt)

        # If we don't care about the labels and just want to compute the metric based on the segmentation, we 
        # mimic same labels
        if pred_label is None:
            gt_label = [1]*len(gt_start_frame)
            pred_label = [1]*len(pred_start_frame)

        for overlap in overlap_list:
            tp = 0
            fp = 0

            hits = np.zeros(len(gt_label))

            for j in range(len(pred_label)):            
                intersection = np.minimum(pred_stop_frame[j], gt_stop_frame) - np.maximum(pred_start_frame[j], gt_start_frame)
                union = np.maximum(pred_stop_frame[j], gt_stop_frame) - np.minimum(pred_start_frame[j], gt_start_frame)
                IoU = (1.0*intersection / union)*([pred_label[j] == gt_label[x] for x in range(len(gt_label))])
                # Get the best scoring segment
                idx = np.array(IoU).argmax()

                if IoU[idx] >= overlap and not hits[idx]:
                    tp += 1
                    hits[idx] = 1
                else:
                    fp += 1
            fn = len(gt_label) - sum(hits)
            
            precision = tp / (len(pred_label) - 1)
            recall = tp / (len(gt_label) - 1)
            f1 = 2.0 * (precision*recall) / (precision+recall)
            
            self.scores['f1']['overlap'].append(overlap)
            self.scores['f1']['tp'].append(float(tp))
            self.scores['f1']['fp'].append(float(fp))
            self.scores['f1']['fn'].append( float(fn))
            self.scores['f1']['precision'].append(precision)
            self.scores['f1']['recall'].append(recall)
            self.scores['f1']['f1'].append(f1)
            
        
        return pd.DataFrame.from_dict(self.scores['f1'])


    def compute_timestamps_f1(self, gt_cpt=None, pred_cpt=None):
            
        """Calculate the precision/recall of an estimated segmentation compared
        with the true segmentation.

        Args:
            gt_cpt (list): list of the last index of each regime (true
                partition).
            pred_cpt (list): list of the last index of each regime (computed
                partition).
            margin (int, optional): allowed error (in points).

        Returns:
            tuple: (precision, recall)
        """
        
        from itertools import product
        sanity_check(pred_cpt, gt_cpt)
        
        assert self.margin > 0, "Margin of error must be positive (margin = {})".format(self.margin)

        if len(pred_cpt) == 1:
            return 0, 0

        used = set()
        true_pos = set(
            true_b
            for true_b, my_b in product(gt_cpt[:-1], pred_cpt[:-1])
            if my_b - self.margin < true_b < my_b + self.margin
            and not (my_b in used or used.add(my_b))
        )

        tp_ = len(true_pos)
        precision = tp_ / (len(pred_cpt) - 1)
        recall = tp_ / (len(gt_cpt) - 1)
        f1 = 2.0 * (precision*recall) / (precision+recall)
        self.scores['timestamps_f1'] = {'tp': float(tp_), 'precision': precision, 'recall': recall, 'f1': f1}
        return self.scores['timestamps_f1']
        
    @staticmethod
    def decompose(cpt):
        from itertools import tee
        def pairwise(iterable):
            """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)


        p_start  = [0] + [a for a, _ in pairwise(cpt)]
        p_end  = [cpt[0]] + [b for _, b in pairwise(cpt)]
        return p_start, p_end
    
    
    
# 3) Build model 
def plot_clustering_metrics(results, hue='model_name'):
    """
    Visualizes the distribution of clustering metrics across 'n_clusters',
    with different 'model_name' values as hues and separate plots per 'cluster_type'.
    """
    metrics = ['silhouette_score_baseline', 'silhouette_score_M', 'silhouette_score_T', 'silhouette_score_X', 
               'davies_bouldin_baseline', 'davies_bouldin_M', 'davies_bouldin_T', 'davies_bouldin_X', 
               'compactness_separation_M', 'compactness_separation_T', 'compactness_separation_X', 
               'mmd_baseline', 'mmd_M', 'mmd_T', 'mmd_X', 
               'modularity_baseline', 'modularity_M', 'modularity_T', 'modularity_X',
               'mean_ic', 'median_ic']
    
    
    gresults_K = results[results['input_space'] == 'K_space'].groupby(['cluster_type', 'temporal_distance', 'input_space', 'model_name', 'linkage_method',  'kernel_name', 'multimodal_method', 'alpha', 'n_clusters'])[
                    [
                        'silhouette_score_baseline', 'silhouette_score_M', 'silhouette_score_T', 'silhouette_score_X', 
                        'davies_bouldin_baseline', 'davies_bouldin_M', 'davies_bouldin_T', 'davies_bouldin_X', 
                        'compactness_separation_baseline', 'compactness_separation_M', 'compactness_separation_T', 'compactness_separation_X', 
                        'mmd_baseline', 'mmd_M', 'mmd_T', 'mmd_X', 
                        'modularity_baseline', 'modularity_M', 'modularity_T', 'modularity_X',
                        'mean_ic', 'median_ic',
                        'original_cluster_index', 
                        'perm', 
                        'labels'
                        
                        ]
                    ].agg(
                    {
                    'silhouette_score_baseline': ['mean', 'std'],
                    'silhouette_score_M': ['mean', 'std'],
                    'silhouette_score_T': ['mean', 'std'],
                    'silhouette_score_X': ['mean', 'std'],
                    'davies_bouldin_baseline': ['mean', 'std'],
                    'davies_bouldin_M': ['mean', 'std'],
                    'davies_bouldin_T': ['mean', 'std'],
                    'davies_bouldin_X': ['mean', 'std'],
                    'compactness_separation_baseline': ['mean', 'std'],
                    'compactness_separation_M': ['mean', 'std'],
                    'compactness_separation_T': ['mean', 'std'],
                    'compactness_separation_X': ['mean', 'std'],
                    'mmd_baseline': ['mean', 'std'],
                    'mmd_M': ['mean', 'std'],
                    'mmd_T': ['mean', 'std'],
                    'mmd_X': ['mean', 'std'],
                    'modularity_baseline': ['mean', 'std'], 
                    'modularity_M': ['mean', 'std'],
                    'modularity_T': ['mean', 'std'],
                    'modularity_X': ['mean', 'std'],
                    'mean_ic': ['mean', 'std'],
                    'median_ic': ['mean', 'std'],
                    'original_cluster_index': 'first',
                    'perm': 'first',
                    'labels': 'first'
                    }).sort_values(('silhouette_score_baseline', 'mean'), ascending=False)
    gresults_G = results[results['input_space'] == 'G_space'].groupby(['cluster_type', 'temporal_distance', 'input_space', 'model_name', 'linkage_method',  'kernel_name', 'multimodal_method', 'alpha', 'n_clusters'])[
                        [
                            'silhouette_score_baseline', 'silhouette_score_M', 'silhouette_score_T', 'silhouette_score_X', 
                            'davies_bouldin_baseline', 'davies_bouldin_M', 'davies_bouldin_T', 'davies_bouldin_X', 
                            'compactness_separation_baseline', 'compactness_separation_M', 'compactness_separation_T', 'compactness_separation_X', 
                            'mmd_baseline', 'mmd_M', 'mmd_T', 'mmd_X', 
                            'modularity_baseline', 'modularity_M', 'modularity_T', 'modularity_X',
                            'mean_ic', 'median_ic',
                            'original_cluster_index', 
                            'perm', 
                            'labels'
                            
                            ]
                        ].agg(
                        {
                        'silhouette_score_baseline': ['mean', 'std'],
                        'silhouette_score_M': ['mean', 'std'],
                        'silhouette_score_T': ['mean', 'std'],
                        'silhouette_score_X': ['mean', 'std'],
                        'davies_bouldin_baseline': ['mean', 'std'],
                        'davies_bouldin_M': ['mean', 'std'],
                        'davies_bouldin_T': ['mean', 'std'],
                        'davies_bouldin_X': ['mean', 'std'],
                        'compactness_separation_baseline': ['mean', 'std'],
                        'compactness_separation_M': ['mean', 'std'],
                        'compactness_separation_T': ['mean', 'std'],
                        'compactness_separation_X': ['mean', 'std'],
                        'mmd_baseline': ['mean', 'std'],
                        'mmd_M': ['mean', 'std'],
                        'mmd_T': ['mean', 'std'],
                        'mmd_X': ['mean', 'std'],
                        'modularity_baseline': ['mean', 'std'], 
                        'modularity_M': ['mean', 'std'],
                        'modularity_T': ['mean', 'std'],
                        'modularity_X': ['mean', 'std'],
                        'mean_ic': ['mean', 'std'],
                        'median_ic': ['mean', 'std'],
                        'original_cluster_index': 'first',
                        'perm': 'first',
                        'labels': 'first'
                        }).sort_values(('silhouette_score_baseline', 'mean'), ascending=False)
                        
    display(gresults_K.head(1))
    display(gresults_G.head(1))



    
    if results.input_space.nunique() == 1:
        
        hue = "alpha"
        cluster_types = results["cluster_type"].unique()
        for cluster_type in cluster_types:
            subset = results[results["cluster_type"] == cluster_type]
            fig, axes = plt.subplots(len(metrics) // 4 +1 , 4, figsize=(25, 16))
            fig.suptitle(f"Clustering Metrics Distribution for Cluster Type: {cluster_type}", fontsize=16)
            
            for j, ( ax, metric) in enumerate(zip(axes.flatten(), metrics)):
                
                # if j in [4, 8]:
                #     ax.set_visible(False)
                #     continue
                #sns.boxplot(data=subset, x="n_clusters", y=metric, hue="model_name", ax=ax)
                subset["combo"] = subset["multimodal_method"] + "_" + subset["kernel_name"]

                if len(subset[hue].unique()) == 1:
                    sns.lineplot(data=subset, x="n_clusters", y=metric, hue=hue, 
                                    ax=ax, marker="o", markersize=8)
                else:
                    #sns.pointplot(data=subset, x="n_clusters", y=metric, hue='combo', 
                    #              ax=ax, errorbar=("sd", 1), dodge=True, markers="o", capsize=0.1)
                    
                    sns.lineplot(
                        data=subset, x="n_clusters", y=metric,
                        hue='alpha', style="kernel_name", 
                        ax=ax, marker="o"
                    )
                # sns.lineplot(            
                    # sns.pointplot(
                    # data=subset, x="n_clusters", y=metric,
                    # hue='combo', style='kernel_name', 
                    # ax=ax, errorbar=("sd", 1), dodge=True, markers="o", capsize=0.1)
                    

                ax.set_title(metric.replace("_", " ").title())
                ax.set_xlabel("Number of Clusters")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.legend(title="Model Name")
                ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
                
    else:
        
        
                    
        metric_groups = ['silhouette_score']#, 'modularity']#, 'mmd', 'compactness_separation']
        declinations = ['baseline', 'M', 'T', 'X']
        melted = results.melt(
            id_vars=['n_clusters', 'cluster_type', 'model_name', 'kernel_name', 'multimodal_method', 'input_space', 'linkage_method', 'alpha'],
            var_name='metric', value_name='value'
        )
        melted = melted[melted['metric'].str.contains('|'.join(metric_groups))]
        melted['metric_group'] = melted['metric'].str.extract(f"^({'|'.join(metric_groups)})")
        melted['declination'] = melted['metric'].str.extract(f"({'|'.join(['baseline', 'M', 'T', 'X'])})$") 
        melted['declination_test'] = melted['multimodal_method'] + " | " + melted['kernel_name']
        melted['declinations_model_name'] = melted['model_name'] + " | " + melted['linkage_method'].fillna('without linkage')

        for metric in metric_groups:
            for ct in melted['cluster_type'].unique():
                for input_space in ['K_space', 'G_space']:
                    data_ct = melted[(melted['metric_group'] == metric) & (melted['cluster_type'] == ct) & (melted['input_space'] == input_space)]

                    g = sns.relplot(
                        data=data_ct,
                        kind="line",
                        x="n_clusters", y="value",
                        hue="declinations_model_name", style="alpha",
                        row="declination_test", col="declination",
                        linewidth=2.5, markers=True, dashes=True,
                        facet_kws={'sharey': False},
                        height=3.5, aspect=1.8
                    )
                    g.set_titles(row_template="{row_name}", col_template="{col_name}")
                    g.set_axis_labels("Number of Clusters", metric)
                    g.fig.subplots_adjust(top=0.9)
                    g.fig.suptitle(f"{metric} | {input_space} | {ct}", fontsize=15)
        plt.tight_layout()
        plt.show()

            



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

def levenstein(p, y, norm=False):
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

def get_predictions_with_label(prediction_by_label, cidx):
    """Get all predicitons of the given label. Return empty DataFrame if there
    is no predcitions with the given label. 
    """
    try:
        return prediction_by_label.get_group(cidx).reset_index(drop=True)
    except:
        #print('Warning: No predictions of label \'%s\' were provdied.' % cidx)
        return pd.DataFrame()
    
def compute_map(pred, gt, overlaps, fps, video_name):

    labels_pred, p_start, p_end = get_labels_start_end_time(pred, bg_class=[0])
    labels_gt, p_start_gt, p_end_gt = get_labels_start_end_time(gt, bg_class=[0])

    prediction = pd.DataFrame({'video-id':[video_name for _ in range(len(labels_pred))],
                               'label' : [label for label in labels_pred],
                               't-start' : [t/fps for t in p_start],
                               't-end' : [t/fps for t in p_end],
                               'score' : [1 for _ in range(len(labels_pred))]})

    ground_truth = pd.DataFrame({'video-id':[video_name for _ in range(len(labels_gt))],
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
    
    return average_mAP

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

def compute_scores(frame_prediction, frame_gt, overlaps, fps, video_name):
    
    labels_pred, _, _ = get_labels_start_end_time(frame_prediction, bg_class=[0])
    labels_gt, _, _ = get_labels_start_end_time(frame_gt, bg_class=[0])

    n_clusters = len(np.unique(labels_gt))
    edit = levenstein(labels_pred, labels_gt, norm=True)

    # Compute accuracy
    correct = 0
    total = 0
    for i in range(len(frame_gt)):
        total += 1
        if frame_gt[i] == frame_prediction[i]:
            correct += 1
    acc = 100 * float(correct) / total


    frame_f1_macro = f1_score(frame_gt, frame_prediction, average='macro')  # F1-Score
    # iou_macro = metrics.jaccard_score(gt_labels, y_pred, average='macro')  # IOU
    # penalize equally over/under clustering
    jaccard = np.sum(jaccard_score(frame_gt, frame_prediction, average=None)) / n_clusters


    # Compute mAP
    mAP = compute_map(frame_prediction, frame_gt, overlaps, fps, video_name)

    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    # Add tp, fp and fn for f1 computation
    for s in range(len(overlaps)):
        tp1, fp1, fn1 = f_score(frame_prediction, frame_gt, overlaps[s])
        tp[s] += tp1
        fp[s] += fp1
        fn[s] += fn1
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlaps)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
    #         print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1

    return acc, frame_f1_macro, jaccard, edit, mAP, f1s[0], f1s[1], f1s[2]


def add_twfinch(pipeline, n_clusters):
    # LEGACY: External benchmark function (TWFINCH), not currently functional.
    # Hardcoded paths to /home/perochon/ — kept for reference only.
    from python.twfinch import FINCH
    from scipy.optimize import linear_sum_assignment
    from sklearn import metrics


    if pipeline.dataset.task == 'Breakfast':
        activity_name = pipeline.dataset.video_name.split('_')[-1]
        cur_filename = os.path.join('/home/perochon/temporal_segmentation/data/Action_Segmentation_Datasets/', pipeline.dataset.task, 'features',activity_name,  pipeline.dataset.video_name + '.txt') 

    else:
        cur_filename = os.path.join('/home/perochon/temporal_segmentation/data/Action_Segmentation_Datasets/', pipeline.dataset.task, 'features', pipeline.dataset.video_name + '.txt')
        activity_name = os.path.basename(os.path.split(cur_filename)[0])

    # Load the Descriptor       
    if os.path.isfile(cur_filename):
        emb = np.loadtxt(cur_filename, dtype='float32')
    else:
        emb = deepcopy(pipeline.embedding.T)



    # Load the Descriptor        
    c, num_clust, prediction_finch = FINCH(emb, req_clust=n_clusters, verbose=False, tw_finch=True)

    if pipeline.dataset.task in ['GTEA', 'EPIC-KITCHEN']:
        cpt= [0] + list(np.argwhere(np.abs(np.ediff1d(prediction_finch))>0).flatten())

        cpt_frame = np.array( (list(np.array(cpt)*pipeline.tau_sample) + [pipeline.dataset.n_frames])).astype(int)

        counter_segments = 0

        segment_label = [prediction_finch[0]]
        count = 0
        for idx, i  in enumerate(prediction_finch[1:]):
            if i != segment_label[count]:
                segment_label.append(i)
                count+=1
                
        pred_frames_label = np.zeros(pipeline.dataset.n_frames)

        for i in range(0, pipeline.dataset.n_frames):

            # Is there a chnage of segment ? 
            if i == cpt_frame[1:][counter_segments]:

                counter_segments+=1    

            pred_frames_label[i] = segment_label[counter_segments]
        pred_frames_label = pred_frames_label.astype(int)
    else:
        pred_frames_label = prediction_finch
                    
        
    if not isinstance(pred_frames_label, list):
        pred_frames_label = list(pred_frames_label)

    if pipeline.annotation.gt_label is not None and len(pred_frames_label) >= len(pipeline.annotation.gt_label):
        pred_frames_label = (np.array(pred_frames_label)[:len(pipeline.annotation.gt_label)]).astype(int)
    elif pipeline.annotation.gt_label is not None:
        pred_frames_label = np.concatenate([pred_frames_label + [pred_frames_label[-1]] * (len(pipeline.annotation.gt_label)-len(pred_frames_label))], axis=0).astype(int)

    else:
        pred_frames_label = np.array(pred_frames_label).astype(int)

    # Find best assignment through Hungarian Method
    cost_matrix = estimate_cost_matrix(pipeline.annotation.gt_label, pred_frames_label)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # decode the predicted labels
    y_pred_tw_finch = col_ind[pred_frames_label]

    predicted_actions = (y_pred_tw_finch!=0).astype(int)
    gt_label = deepcopy(pipeline.annotation.gt_label)
    gt_label[np.argwhere(np.abs(np.ediff1d(gt_label))>0).flatten()] = 0
    gt_actions = (gt_label!=0).astype(int)
    f1_actions_post = pipeline._compute_segmental_f1(predicted_actions,  gt_actions, overlap=.25, verbose=False)
            
    return y_pred_tw_finch, f1_actions_post

def save_experiment_metrics(df, path, pipeline, name, granularity='action', acc=None, frame_f1_macro=None, jaccard=None, edit=None, mAP=None, map_10=None, map_20=None, map_30=None, map_40=None, map_50=None,  f1_10=None, f1_25=None, f1_50=None, acc_tw=None, frame_f1_macro_tw=None, jaccard_tw=None, edit_tw=None, mAP_tw=None, map_10_tw=None, map_20_tw=None, map_30_tw=None, map_40_tw=None, map_50_tw=None,f1_10_tw=None, f1_25_tw=None, f1_50_tw=None, f1_actions_post_tw=None):
    
    df = df.append(pd.DataFrame({'dataset': pipeline.dataset.task,
                                'video_name': pipeline.dataset.video_name, 
                                'n_frames': pipeline.dataset.n_frames, 
                                'name': name, 
                                'granularity': granularity,
                                'penalty': pipeline.penalty,
                                'f1_actions': pipeline.f1_actions,
                                 'f1_actions_post' : pipeline.f1_actions_post,
                                 'f1_actions_post_tw': f1_actions_post_tw,
                                 'cpt': len(pipeline.cpt),
                                 'pred_bk': np.mean(pipeline.annotation.pred_frames_label==0),
                                 'gt_bk': np.mean(pipeline.annotation.gt_label==0),
                                'acc': acc, 
                                'acc_tw': acc_tw, 
                                'frame_f1_macro': frame_f1_macro, 
                                'frame_f1_macro_tw': frame_f1_macro_tw, 
                                'jaccard': jaccard, 
                                'jaccard_tw': jaccard_tw, 
                                'edit': edit, 
                                'edit_tw': edit_tw, 
                                'mAP': mAP, 
                                'map_10': map_10, 
                                'map_20': map_20, 
                                'map_30': map_30, 
                                'map_40': map_40, 
                                'map_50': map_50, 
                                'mAP_tw': mAP_tw, 
                                'map_10_tw': map_10_tw, 
                                'map_20_tw': map_20_tw, 
                                'map_30_tw': map_30_tw, 
                                'map_40_tw': map_40_tw, 
                                'map_50_tw': map_50_tw,
                                'f1_10': f1_10, 
                                'f1_25': f1_25, 
                                'f1_50': f1_50, 
                                'f1_10_tw': f1_10_tw, 
                                'f1_25_tw': f1_25_tw, 
                                'f1_50_tw': f1_50_tw}, index=[0]), ignore_index=True)

    df.to_csv(path, index=False)
    #display(df.tail(1))
    return df

# def init_df_metrics(path=os.path.join(DATA_DIR,'results.csv')):
    

    if not os.path.isfile(path):
        results_dict = {'dataset':[],
                        'video_name': [],
                        'n_frames': [],
                        'name': [],
                        'granularity': [],
                        'penalty' : [],
                        'f1_actions': [], 
                        'f1_actions_post': [],
                        'f1_actions_post_tw': [],
                        'cpt': [],
                        'pred_bk': [],
                        'gt_bk': [],
                        'acc': [],
                        'acc_tw': [],
                        'frame_f1_macro': [],
                        'frame_f1_macro_tw':[],
                        'jaccard':[],
                        'jaccard_tw': [], 
                        'edit': [], 
                        'edit_tw': [], 
                        'mAP': [], 
                        'map_10': [], 
                        'map_20': [], 
                        'map_30': [], 
                        'map_40': [], 
                        'map_50': [], 
                        'mAP_tw': [], 
                        'map_10_tw': [], 
                        'map_20_tw': [], 
                        'map_30_tw': [], 
                        'map_40_tw': [], 
                        'map_50_tw': [],
                        'f1_10': [], 
                        'f1_25': [], 
                        'f1_50': [], 
                        'f1_10_tw': [], 
                        'f1_25_tw': [], 
                        'f1_50_tw': []
                        }        
        df = pd.DataFrame(results_dict)

    else:

        df = pd.read_csv(path)
        results_dict = df.to_dict(orient='list')
        
    return df