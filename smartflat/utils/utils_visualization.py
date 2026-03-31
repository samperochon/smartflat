
#import pandas as pd
import datetime
import os
import time
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#import umap.umap_ as umap
from IPython.display import clear_output, display
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler

from smartflat.annotation_smartflat import get_annotation_constants
from smartflat.utils.utils import pad_sequence_with_zeros, upsample_sequence
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_dataset import get_long_embedding
from smartflat.utils.utils_io import parse_identifier
from PIL import Image as PILImage

#COLORS = mpl.colormaps['tab20'](range(20))
    
    
def plot_1d(df, col1, col2, ax=None):

    ax.scatter(df[col1], df[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    return ax
    
def plot_subplots_identifier(results, col1 = 'penalty', col2 = 'n_cpts', fname=None):
    """Plot for each identifier of a result dataframe the 2D p plots of col1 and col2."""
    
    # Obtenir la liste unique des identifiants
    identifiers = results['identifier'].unique()
    results = results.sort_values('penalty')
    # Déterminer le nombre de lignes et de colonnes pour les sous-graphiques
    n_cols = 4
    n_rows = int(np.ceil(len(identifiers) / n_cols))
    
    # Créer une figure et un ensemble de sous-graphiques
    fig, axs = plt.subplots(n_rows, n_cols, sharex=False, figsize=(25, 5*n_rows))
    
    # Créer un graphique pour chaque identifiant
    for i, identifier in enumerate(identifiers):
        # Filtrer le DataFrame pour ne garder que les lignes correspondant à l'identifiant actuel
        df_id = results[results['identifier'] == identifier].copy()
        try:
            participant_id, task_name, modality, _ = parse_identifier(identifier)
        except:
            axs[row, col].axis('off')
            continue        
        row = i // n_cols
        col = i % n_cols

        # FOr this type of plot. 
        df_id.sort_values(col1, inplace=True)
        axs[row, col] = plot_1d(df_id, col1, col2, ax=axs[row, col])
        axs[row, col].set_title(f'{participant_id} - {modality}')

    # Supprimer les axes vides
    for i in range(len(identifiers), n_rows*n_cols):
        fig.delaxes(axs.flatten()[i])
    
    # Afficher la figure
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname=fname)
    plt.show()

def print_get_last_modified_date(file_path):
    '''Print the last modification date of a file.'''
    timestamp = os.path.getmtime(file_path)
    date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Last modification of {os.path.basename(file_path)}: {date}')

def plot_chronogames(df, labels_col='embedding_labels', n_subjects=None, idx=None, n_t_max=None, 
                     max_duration=None, upsampling='interpolation', ax=None, n_colors=None,
                     title='', figsize=(25, 3), time_calibration=None, use_codebook=False, 
                     use_colorbar=True, cmap=None, norm=None, boundaries=None, mapping_labels_int=None,
                     perceptually_consistent=False):
    
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    
    if max_duration is None:
        max_duration = df[labels_col].apply(len).max()

    if type(df[labels_col].iloc[0][0]) in  [str, np.str_]:
        
        print('Mapping string labels to integers for visualization')
        if mapping_labels_int is None:
            mapping_labels_int = {l: i for i, l in enumerate(np.unique(np.hstack(df[labels_col])))}; #mapping_labels_int['-1'] = -1
        
        df[f'int_{labels_col}'] = df[labels_col].apply(lambda x: np.array([mapping_labels_int[i] for i in x]))
        labels_col = f'int_{labels_col}'


    # Upsample labels based on specified method
    if upsampling == 'interpolation':
        upsampled_labels = df[labels_col].dropna().apply(upsample_sequence, args=(max_duration,))
    elif upsampling == 'padding':
        upsampled_labels = df[labels_col].dropna().apply(pad_sequence_with_zeros, args=(max_duration,))
    else:
        raise ValueError('Upsampling method not recognized, use "interpolation" or "padding"')
    
    Y = np.vstack(upsampled_labels.apply(np.array).to_list())
    
    # Get unique class labels and create a colormap
    if mapping_labels_int is not None:
        unique_labels = np.sort(np.unique(list(mapping_labels_int.values())))
    else:
        unique_labels = np.sort(np.unique(Y.astype(int)))
    n_labels = len(unique_labels)

    if 'labels' in labels_col:

        if perceptually_consistent:

            # Ensure colors match unique labels in Y
            present_labels = np.unique(Y)
            colors = sns.color_palette("hls", len(present_labels))
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(boundaries=np.arange(len(present_labels) + 1) - 0.5, ncolors=len(present_labels))

            # Apply the colormap to the data
            colored_matrix = cmap(norm(Y))

        elif cmap is None and norm is None:

            # Get unique class labels and create a colormap
            unique_labels = np.sort(np.unique(Y.astype(int)))
            n_labels = len(unique_labels)
            edges_color = (0.88, 0.93, 0.97, 1)  # Pale Blue
            edges_color = (1, 1, 1, 0)  # Pale Blue
            noise_color = (0, 0, 0, 1) # Black
            print(f'Label range: [{unique_labels.min()}-{unique_labels.max()}] ({n_labels} unique labels)')
            if n_colors is None:
                n_colors = n_labels + 1
            #print('Using {} colors'.format(n_labels))
            colors = get_base_colors(n_colors=n_colors, verbose=False)
            #print(f'Using {len(colors)} colors')
            #color_space = np.sort(np.unique(np.extend(unique_labels, [-2, -1])))  # Add -1 for noise
            color_space = np.sort(np.unique(unique_labels))#np.concatenate([unique_labels, [-2, -1]])))
            color_mapping = {label: colors[i] for i, label in enumerate(color_space)}
            color_mapping[0] = noise_color       # Noise            
            cmap = ListedColormap([color_mapping[label] for label in color_space])
            boundaries = np.append(unique_labels, np.max(color_space)+1)
            norm = BoundaryNorm(boundaries, cmap.N, clip=False)
    
    else:
        
        # For continuous labels, use a continuous colormap like Viridis
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(vmin=np.min(Y), vmax=np.max(Y))
        colored_matrix = cmap(norm(Y))
        
    # If ax is None, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if idx is None and n_subjects is not None:
        idx = np.random.choice(Y.shape[0], size=n_subjects, replace=False)
    else:
        if idx is None:
            idx = np.arange(Y.shape[0])
    im = ax.imshow(Y[idx, :n_t_max], interpolation='nearest', aspect='auto', cmap=cmap, norm=norm)
    ax.set_xlabel('Time [min]'); ax.set_ylabel('Label ID')
    
    if df.identifier.nunique() == 1:
        title = title+ f'{df.identifier.iloc[0]} - {df.modality.iloc[0]} \nChronograms using {labels_col}, {n_labels} label used'
        ax.set_title(title, weight='bold')
    else:
        ax.set_title(title + f'\n{df.identifier.nunique()} participants\n{labels_col}',  weight='bold')

    ax.get_yaxis().set_visible(False)
    
    # Add colorbar with labels
    if use_codebook:
        
        _, cuisine_mapping_dict, _, _, _  = get_annotation_constants()
        mapping_code_semantic = pd.DataFrame(cuisine_mapping_dict).T.reset_index().groupby(['code']).semantic.agg('first').to_dict()
        mapping_code_semantic[-1] = 'N.A: Off labels'
        tick_positions = (boundaries[:-1] + boundaries[1:]) / 2

        # Add centered colorbar labels
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, boundaries=boundaries, ticks=tick_positions)
        cbar.ax.set_yticklabels([mapping_code_semantic.get(label, '') for label in present_labels])


    # --- Colorbar with original labels ---
    if mapping_labels_int is not None:
        print('dffds')
        reverse_mapping = {v: k for k, v in mapping_labels_int.items()}  # int -> original label
        reverse_mapping[0] = 'Background'
        tick_positions = (boundaries[:-1] + boundaries[1:]) / 2
        tick_labels = [reverse_mapping[int(t)] for t in unique_labels]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        try:
            cbar = plt.colorbar(sm, ax=ax, boundaries=boundaries, ticks=tick_positions)
        except:
            cbar = plt.colorbar(sm, ax=ax, boundaries=boundaries, ticks=np.arange(len(tick_labels)))
        cbar.ax.set_yticklabels(tick_labels)
        
    if ('labels' not in labels_col) and use_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)

    # Optionally apply time_calibration function to xtick labels
    if time_calibration == 'frame':
        assert df.identifier.nunique() == 1, "Only support single-chronogram calibration for now"
        def time_calibration_func(x):
            return int((x / df['fps'].iloc[0]) / 60)
        xticks = ax.get_xticks()
        xtick_labels = [time_calibration_func(x) for x in xticks]
        ax.set_xticklabels(xtick_labels)
    
    elif time_calibration == 'embeddings':
        xtick_labels = [np.round(x * 8 / 25 / 60, 2) for x in  ax.get_xticks()]
        ax.set_xticklabels(xtick_labels)
    
    # Show the plot only if ax was not passed
    #plt.show()

    return idx, cmap, norm, boundaries, mapping_labels_int#ax #ax  # Return the axis for further manipulation if needed

def get_base_colors(n_colors, verbose=False):
    colormaps = [
        plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c, 
        plt.cm.Set1, plt.cm.Set2, plt.cm.Set3, plt.cm.Accent, plt.cm.Pastel1, plt.cm.Pastel2
    ]

    # Collect colors from the colormaps
    base_colors = []
    for cmap in colormaps:
        base_colors.extend([cmap(i % cmap.N) for i in range(cmap.N)])

    n_max_colors = len(base_colors)
    
    if n_colors > n_max_colors:
        print(f"Warning: Requested {n_colors} colors, but only {n_max_colors} are available. Repeating the color pool.")
        
    # If not enough colors, repeat the pool
    while len(base_colors) < n_colors:
        base_colors += base_colors

    # Trim to desired length
    base_colors = base_colors[:n_colors]

    if verbose:
        # Display the colors
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.imshow([base_colors], extent=[0, n_colors, 0, 1], aspect='auto')
        ax.set_title(f"{n_colors} categorical colors")
        ax.set_yticks([])
        plt.show()
        
    return base_colors

def plot_labels_2D_encoding(row, xlim=None, labels_col='segments_labels', ax=None, do_order_labels=True, perceptually_consistent=True, figsize=(35, 2)):
    
    if type(row) == pd.DataFrame:
        row = row.iloc[0]
        
    label_sequence = row[labels_col]
    if xlim is not None:
        label_sequence = label_sequence[xlim[0]:xlim[1]]
   
    unique_labels = np.sort(np.unique(label_sequence))

    if do_order_labels:
        # Order labels by first appearance
        label_order = sorted(unique_labels, key=lambda label: np.argmax(label_sequence == label))
        y_axis_label = "Label values ordered by increasing first appeareance timestamps"
    else:
        label_order = unique_labels
        y_axis_label = "Label values"
        
    
    # Create a label matrix and assign label values to each row
    label_matrix = np.zeros((len(label_order), len(label_sequence)))
    for i, label in enumerate(label_order):
        label_matrix[i, :] = np.where(label_sequence == label, label, np.nan)

    #label_matrix[np.isnan(label_matrix)] = -2

    # Consistent colormap logic
    n_labels = len(label_order)
    
    if perceptually_consistent:

        # Ensure colors match unique labels in Y
        
        colors = sns.color_palette("hls", len(unique_labels))
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries=np.arange(len(unique_labels) + 1) - 0.5, ncolors=len(unique_labels))
        # Apply the colormap to the data
        colored_matrix = cmap(norm(label_matrix))

    else:
        # Get unique class labels and create a colormap
        edges_color = (0.88, 0.93, 0.97, 1)  # Pale Blue
        noise_color = (0, 0, 0, 1) # Black
        
        print(f'Label range: [{unique_labels.min()}:{unique_labels.max()}] ({n_labels} unique labels)')
        colors = get_base_colors(n_colors=n_labels+1, verbose=False)
        color_space = np.sort(np.unique(unique_labels))#np.concatenate([unique_labels, [-2, -1]])))
        color_mapping = {label: colors[i] for i, label in enumerate(color_space)}
        color_mapping[-2] = edges_color  # Background
        color_mapping[-1] = noise_color       # Noise     
        # Map colors to matrix
        cmap = ListedColormap([color_mapping[label] for label in unique_labels])
        norm = plt.Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))

            
    title_info = f'{row.task_name} - {row.participant_id} - {row.modality} - support K={len(np.unique(unique_labels))} clusters'

    # Display result
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        label_matrix,
        aspect="auto",
        cmap=cmap,
        norm=norm,  
        interpolation="nearest",
    )
    ax.set_title(title_info, weight='bold', fontsize=12)
    ax.set_ylabel('', weight="bold")
    ax.set_xlabel("Frames", weight="bold")
    ax.set_yticks([])

    #ax.colorbar(label="Label ID")
    #plt.show()

    return label_matrix

def add_color(df, group_col='participant_id', gradient_col='index'):

    def get_color(row):
        # Generate a light palette for the participant
        cmap = sns.light_palette(palette[row['participant_id_enc']], as_cmap=True)
        
        # Get a color from the palette based on 'time_scaled'
        color = cmap(row['time'])
        
        return color

    palette = sns.color_palette('husl', df[group_col].nunique())
    df[f'{group_col}_enc'] = df[group_col].astype('category').cat.codes

    if gradient_col == 'index':
        df['time_original'] = df.index; 
        df['time'] = df.groupby(group_col)['time_original'].transform(lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1,1))[:, 0])
        df['color'] = df.apply(get_color, axis=1)

    return df
    

def get_cmap(all_labels):
    
    n_colors = len(all_labels) + 1
    base_colors = get_base_colors(n_colors=n_colors)
    # Reserved colors
    edges_color = (1, 1, 1, 0)  # Transparent
    noise_color = (0, 0, 0, 1)  # Black
    color_space = np.sort(np.unique(np.concatenate([all_labels, [-1]])))
    color_mapping = {label: base_colors[i] for i, label in enumerate(color_space)}
    color_mapping[-2] = edges_color
    color_mapping[-1] = noise_color
    #cmap = ListedColormap(color_mapping)
    
    # Create a list of colors in the correct order for ListedColormap
    colors = [color_mapping[label] for label in color_space]
    cmap = ListedColormap(colors)
    return cmap


def plot_umap_participants(dset, group_col='participant_id', n = 3):
    
    from copy import deepcopy
    stored_metadata = deepcopy(dset.metadata)
    samples = dset.metadata[group_col].sample(n).tolist()
    dset.metadata = dset.metadata[dset.metadata[group_col].isin(samples)].copy()
    df = get_long_embedding(dset)
    df = add_umap(df)
    df = add_color(df)
    
    dset.metadata = stored_metadata
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='emb_x1', y='emb_x2', c =df['color'], s=100, legend=True)
    plt.title('UMAP representation')
    plt.show()

def plot_gram(row, temporal_segmentation_col='cpts', segments_labels_col='symb_segments_labels', ax=None, n_min=None, n_max=None, title="", plot_legend=False):
    
    if type(row) == np.ndarray:
        embedding = row
        fi(30, 12)
        plt.imshow(embedding @ embedding.T)
        return 

    
    # Gather the righ slice of the big 1D labels array
    embedding = np.load(row['video_representation_path'])[row.test_bounds[0]:row.test_bounds[1]]

    if 'cpts' in row.keys():
        cpts = row['cpts']
    else:
        cpts = []

    if ax is None:
        fi(30, 12)
        ax=plt.gca()

    ax.imshow(embedding @ embedding.T)
    
    if temporal_segmentation_col in row.keys() and segments_labels_col in row.keys():
        if 'segments_start' not in row.keys():
            row["segments_start"], row["segments_end"] = row[temporal_segmentation_col][:-1], row[temporal_segmentation_col][1:]


    if 'segments_start' in row.keys() and segments_labels_col in row.keys():
        
        Y = np.array(row[segments_labels_col])
        unique_labels = np.unique(Y)  # Map labels to contiguous range

        n_labels = len(np.unique(Y))
        #print(f'Found a total of {n_labels} unique labels for this sample')
        #cmap = plt.cm.get_cmap('tab20b', n_labels)
        #cmap = plt.cm.get_cmap('plasma', n_labels)
        # colors = 20 * list(plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors)
        # colors = get_base_colors(n_colors=n_labels+1, verbose=False)
        # cmap = ListedColormap(colors[:n_labels]) 
        
        # # Normalize the labels for colormap scaling
        # norm = plt.Normalize(vmin=0, vmax=np.max(Y) - 1)
        # colored_matrix = cmap(norm(Y))
        # colored_matrix = cmap(Y / (n_labels - 1))
        
        
        noise_color = (0, 0, 0, 1) # Black
        #noise_color = (1, 1, 1, 1) # White
        
        print(f'Label range: [{unique_labels.min()}:{unique_labels.max()}] ({n_labels} unique labels)')
        
        colors = get_base_colors(n_colors=n_labels+1, verbose=False)
        color_mapping = {label: colors[i] for i, label in enumerate(unique_labels)}
        color_mapping[-2] = noise_color  # Background
        #color_mapping[-1] = noise_color       # Noise

        # Map colors to matrix
        cmap = ListedColormap([color_mapping[label] for label in unique_labels])
        norm = plt.Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))

        colored_matrix = cmap(norm(Y))
            
            
            
            

        labels=[]
        for i, j, label in zip(row['segments_start'],
                                      row['segments_end'], 
                                      row[segments_labels_col]):

            #color = colored_matrix[int(label)]
            mapped_label = np.where(unique_labels == label)[0][0]  # Get the remapped index
            color = colored_matrix[mapped_label]  # Use mapped index for lookup

            ax.fill_betweenx([-0.1*embedding.shape[1], 0], x1=[i, i], x2=[j, j], color=color, alpha=1, label = label if label not in labels and plot_legend else None)
            labels.append(label)
        if plot_legend:
            ax.legend(loc='lower right')

    for cpt in cpts:
        ax.fill_betweenx([-0.2*embedding.shape[1], -0.1*embedding.shape[1]], x1=[cpt, cpt], x2=[cpt+2, cpt+2], color='red')
        #ax.axvline(x=cpt, ymin = -0.2*embedding.shape[1], ymax = -0.09*embedding.shape[1], lw = 3, color = 'red')
        #ax.axvline(x=cpt, lw = 3, color = 'red')
    
    ax.set_title(title, weight='bold', fontsize=20)
   
    # Change xticks label from embeddings indexing to minutes
    #ax.set_xticks(ax.get_xticks(), np.round(ax.get_xticks()*5/25/60, 2).astype(str))
    #ax.set_yticks(ax.get_yticks(), np.round(ax.get_yticks()*5/25/60, 2).astype(str))

    #ax.set_ylim(-0.4*embedding.shape[1], embedding.shape[1])
    #ax.set_xlim(0, embedding.shape[1])
    
    return ax
    
def plot_per_cat_x_cont_y_distributions(df, x_cat='segments_labels', y_cont='segments_fit_cost', plot_samples=True, n_rows=2, n_cols=2, figsize=(25, 3)):
    
    def pool_lengths(length_dicts):
        pooled = defaultdict(list)
        for d in length_dicts:
            for label, lengths in d.items():
                pooled[label].extend(lengths)
        return pooled

    def get_colors(labels):
        cmap = plt.cm.get_cmap('tab20', len(labels))
        norm = plt.Normalize(vmin=0, vmax=len(labels) - 1)
        return [cmap(norm(i)) for i in range(len(labels))]

    # Step 1: Aggregate lengths per cluster for each row
    df['cluster_lengths'] = df.apply(lambda row: pd.Series(row[y_cont]).groupby(row[x_cat]).apply(list).to_dict(), axis=1)
    # Step 2: Pool lengths globally
    global_clusters = pool_lengths(df['cluster_lengths'])

    # Step 3: Plot global boxplots by cluster
    labels = sorted(global_clusters.keys())
    colors = get_colors(labels)

    plt.figure(figsize=figsize)
    # plt.boxplot([global_clusters[label] for label in labels], labels=labels, patch_artist=True,
    #             boxprops=dict(facecolor=colors), showfliers=False)
    
    boxplots = plt.boxplot([global_clusters[label] for label in labels], labels=labels, patch_artist=True, showfliers=False)

    # Set individual box colors
    for patch, color in zip(boxplots['boxes'], colors):
        patch.set_facecolor(color)
        
    
    plt.title(f'x_cat={x_cat} - y_cont={y_cont}', weight='bold')
    plt.xlabel(x_cat)
    plt.ylabel(y_cont)
    plt.show()

    if plot_samples:
        # Step 4: Plot individual distributions

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6), sharex=True, sharey=True)
        fig.suptitle(f'x_cat={x_cat} - y_cont={y_cont}')
        axes = axes.flatten()

        for i, clusters in enumerate(df['cluster_lengths'][:n_rows*n_cols]):
        
            labels, values = zip(*[(k, v) for k, v in clusters.items()])
            for label, value in clusters.items():
                axes[i].boxplot(value, positions=[list(labels).index(label)], patch_artist=True,
                                boxprops=dict(facecolor=colors[labels.index(label)]), 
                                showfliers=False)
            axes[i].set_title(f'Example {i + 1}')
            axes[i].set_xlabel('Clusters')
            axes[i].set_ylabel(y_cont)

        plt.tight_layout()
        plt.show()
    
def plot_2D_with_covariates(df, variable, categorical_columns, continuous_columns, abscisse='duration'):
    # Combine the columns into a single list with categories first
    all_columns = categorical_columns + continuous_columns
    num_plots = len(all_columns)
    
    # Create subplots
    cols = int(np.ceil(num_plots / 3))
    fig, axes = plt.subplots(3, cols, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f'{variable} vs {abscisse}', weight='bold')
    axes = axes.flatten()  # Flatten in case we don't have exactly 2 columns
    for idx, color_col in enumerate(all_columns):
        ax = axes[idx]
        if color_col in categorical_columns:
            # Categorical coloring
            encoded_col = f'{color_col}_col'
            df[encoded_col] = df[color_col].astype('category').cat.codes
            color_labels = dict(enumerate(df[color_col].astype('category').cat.categories))
            
            # Filter the colormap to only use the needed colors
            unique_values = df[encoded_col].unique()
            cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, 10)[unique_values]))
            
            scatter = ax.scatter(
                df[abscisse], 
                df[variable], 
                c=df[encoded_col], 
                cmap=cmap, 
                alpha=0.7
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_ticks(list(color_labels.keys()))
            cbar.set_ticklabels(list(color_labels.values()))
            df.drop(columns=encoded_col, inplace=True)
        else:
            # Continuous coloring
            cmap = 'plasma'
            scatter = ax.scatter(
                df[abscisse], 
                df[variable], 
                c=df[color_col], 
                cmap=cmap, 
                alpha=0.7
            )
            fig.colorbar(scatter, ax=ax)
            
            
        # KDE overlay
        
        xy = np.vstack([df.dropna(subset=[variable, abscisse])[abscisse], df.dropna(subset=[variable, abscisse])[variable].replace(np.inf, 1e8)])
        kde = gaussian_kde(xy, weights=None)
        x, y = np.meshgrid(
            np.linspace(df[abscisse].min(), df[abscisse].max(), 100),
            np.linspace(df[variable].min(), df[variable].replace(np.inf, 1e8).max(), 100)
        )
        positions = np.vstack([x.ravel(), y.ravel()])
        
        kde_values = kde(positions).reshape(x.shape)
        
        ax.contour(
            x, y, kde_values, colors='black', alpha=0.2, linewidths=.5, linestyles='dotted'
        )
            
        
        # Extend limits by 10%
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Titles and labels
        ax.set_title(f'{variable} vs {abscisse} \n(colored by {color_col})')
        ax.set_xlabel(abscisse)
        ax.set_ylabel(f'{variable}')
            
    # Remove any unused axes
    for idx in range(len(all_columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.show()


def dynamic_row_plot_func(df, func, n_sbj=25, delay=0.5, figsize=(25, 5), **kwargs): #labels_col='segments_labels', 
    """
    Dataframe rows dynamic visualization, looping over `identifier` and then `n_clusters`.
    Updating the plot in place.
    
    Parameters:
        data (pd.DataFrame): Data containing the label sequences.
        labels_col (str): Column name containing the label sequences.
        delay (float): Time in seconds to wait between frames.
    """
    
    assert func is not None
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=figsize)
        
    for _ in range(n_sbj): 
    
        identifier = df.identifier.sample(1).values[0]
        
        print(f"Visualizing {identifier}")
        
        row = df[(df.identifier == identifier)]# & (df['n_cluster'] == n_cluster)]
        
        ax.clear()
        func(row, ax=ax, **kwargs)
        clear_output(wait=True)
        display(fig)  # Redisplay the updated figure
        time.sleep(delay)
        
    plt.close()

def plot_qualification_mapping(cluster_dict):
    plt.figure(figsize=(8, 2))
    
    for i, (cluster_name, indices) in enumerate(cluster_dict.items()):
        plt.scatter(indices, [i] * len(indices), label=cluster_name, alpha=0.7)
    
    plt.yticks(range(len(cluster_dict)), cluster_dict.keys())
    plt.title('cluster-type assignement distribution over the prototypes index')
    plt.xlabel("Cluster Index")
    plt.ylabel("Cluster Type")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def plot_multiple_qualification_mapping(clustering_dict, figsize=(20, 10)):
    # num_clusterings = len(clustering_dict)
    # fig, axes = plt.subplots(num_clusterings // 2+1, 2, figsize=figsize); axes = axes.flatten()

    # if num_clusterings == 1:
    #     axes = [axes]  # Ensure axes is iterable when there's only one subplot
    
    # for ax, (title, clusters) in zip(axes, clustering_dict.items()):
    #     cluster_names = list(clusters.keys())  # Ensure consistent ordering
    #     y_positions = list(range(len(cluster_names)))  # Assign y values

    #     for y_pos, cluster_name in zip(y_positions, cluster_names):
    #         indices = clusters[cluster_name]
    #         ax.scatter(indices, [y_pos] * len(indices), label=cluster_name, alpha=0.7)

    #     ax.set_yticks(y_positions)
    #     ax.set_yticklabels(cluster_names)  # Ensures alignment
    #     ax.set_title(title)
    #     ax.legend()  # Uses plotted order
    #     ax.grid(True, linestyle="--", alpha=0.5)
    
    # plt.xlabel("Cluster Index")
    # plt.tight_layout()
    # plt.show()

    n_plot = 0
    valid_configs = []
    for clustering_config_name, q_annotator in clustering_dict.items():
        
        for annotator_id, q_rounds in q_annotator.items():
            
            for round_number, q_cluster_types in q_rounds.items():
                n_plot += 1
                valid_configs.append([clustering_config_name, annotator_id, round_number])


    figsize = (25, 60)

    fig, axes = plt.subplots(n_plot // 2+1, 2, figsize=figsize); axes = axes.flatten()

    if n_plot == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one subplot

    for ax, (clustering_config_name, annotator_id, round_number) in zip(axes, valid_configs):
        clusters = clustering_dict[clustering_config_name][annotator_id][round_number]
        cluster_names = list(clusters.keys())  # Ensure consistent ordering
        y_positions = list(range(len(cluster_names)))  # Assign y values

        for y_pos, cluster_name in zip(y_positions, cluster_names):
            indices = clusters[cluster_name]
            ax.scatter(indices, [y_pos] * len(indices), label=cluster_name, alpha=0.7)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(cluster_names)  # Ensures alignment
        ax.set_title(f'{clustering_config_name}\n{annotator_id} - {round_number}')
        ax.legend()  # Uses plotted order
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.xlabel("Cluster Index")
    plt.tight_layout()
    plt.show()


# Binary variables 
def plot_binary(array, ax=None, title="", names = ['Inlier', 'Outlier']):

    if ax is None:
        fi(25, 3);ax=plt.gca()

    final_title = "Outlier detection binary mask\n" + title

    
    cmap = ListedColormap(['lightgrey', 'k'])
    n=2

    ax = sns.heatmap(array[np.newaxis, :], cmap=cmap, ax=ax)
    # modify colorbar:
    colorbar = ax.collections[0].colorbar 
    r = colorbar.vmax - colorbar.vmin 
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(names)
    ax.set_title(final_title, weight='bold')

    return ax

# Computation-related
def computation_state_report(df):
    
    blue("Assignement per machines:")
    display(df.drop_duplicates(subset= ['identifier']).assignement.value_counts())
    
    blue("Percentage of embedding computed: {}/{} ({:.1f}) %".format(df.drop_duplicates(subset=['identifier'])['video_representation_computed'].sum(), df.drop_duplicates(subset=['identifier'])['video_representation_computed'].shape[0], df.drop_duplicates(subset=['identifier'])['video_representation_computed'].mean()*100))

def plot_state(df):
    counts = df.groupby(['machine_name', 'video_representation_computed']).size().unstack(fill_value=0)

    # Calculate the percentage of embeddings computed for each machine
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    # Plot the counts and percentages
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    counts.plot(kind='bar', stacked=True, ax=ax[0])
    ax[0].set_title('Number of embeddings computed for each machine')
    ax[0].set_ylabel('Count')

    percentages.plot(kind='bar', stacked=True, ax=ax[1])
    ax[1].set_title('Percentage of embeddings computed for each machine')
    ax[1].set_ylabel('Percentage')

    plt.tight_layout()
    plt.show()
    
    
def plot_distance_evolution(df, segments_labels_col='segments_labels', num_rows=10, do_filter=False):
    """
    Plots the evolution of distances for embeddings across multiple rows in the DataFrame, 
    including segments with colors and change-points as vertical lines.

    Parameters:
    - df: DataFrame containing the 'distances', 'segments_labels', and 'cpts' columns.
    - num_rows: Number of rows to plot (default is 10).
    - do_filter: Whether to limit the y-axis for better visibility.
    """
    assert df.n_cluster.nunique() == 1, "All rows must have the same number of clusters"
    
    # Limit to the specified number of rows
    subset = df.iloc[:num_rows]
    
    
    n_cluster = df.n_cluster.iloc[0]    
    colors = 20*list(plt.cm.tab20b.colors + plt.cm.tab20.colors + plt.cm.tab20c.colors)
    
    charcoal = (0.21, 0.27, 0.31, 1)
    gray = (0.5, 0.5, 0.5, 1)           # Medium gray
    light_gray = (0.75, 0.75, 0.75, 1)   # Light gray
    dark_gray = (0.25, 0.25, 0.25, 1)    # Dark gray
        
    
    colors[-1] = light_gray
    
    xmin, xmax = 1500, 4000

    present_labels = np.unique(np.hstack(subset[segments_labels_col]))
    cluster_colors = ListedColormap([colors[label] for label in present_labels])

    # Create subplots
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(25, 10), sharex=False, sharey=True)
    axes = axes.flatten()
    fig.suptitle(f"Distance to first centroid neighbors\n{segments_labels_col}", fontsize=20)

    # Plot distance evolution for each row
    for i, (idx, row) in enumerate(subset.iterrows()):
        distances = row['cluster_dist']
        segments = row[segments_labels_col]
        cpts = row['cpts']

        dict_label_plotted = []
        # Plot the distance signal
        axes[i].plot(distances, color='k', label='Global clustering')
        
        # Add segments with colors
        for j in range(len(cpts) - 1):
            start, end = cpts[j], cpts[j + 1]
            segment_color = cluster_colors(segments[j])
            axes[i].plot(range(start, end), distances[start:end], color=segment_color, linewidth=2)#, label=f'start={start}, end={end} , label={segments[j]}' if segments[j] not in dict_label_plotted else None)
            dict_label_plotted.append(segments[j])
            if (i==0) and (j == 0):
                print(f'start={start}, end={end} , label={segments[j]}')
        
        # Add vertical lines for change-points
        # for cp in cpts:
        #     axes[i].axvline(cp, color='black', linestyle='--', linewidth=0.8)
        
        # Configure plot appearance
        axes[i].set_title(f"{row.identifier}")
        axes[i].set_xlabel("Embedding Index")
        axes[i].set_ylabel("Distance")
        axes[i].legend()
        axes[i].set_xlim(xmin, xmax)
        if do_filter:
            axes[i].set_ylim(0, 0.7)

    # Hide extra subplots
    for ax in axes[len(subset):]:
        ax.axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def make_mixed_images(impaths):
    new_paths = []
    for impath in impaths:
        furthest_path = impath.replace('closest', 'furthest')
        mixed_path = impath.replace('closest', 'mixed')
        os.makedirs(os.path.dirname(mixed_path), exist_ok=True)
        if os.path.exists(furthest_path):
            im1 = PILImage.open(impath)
            im2 = PILImage.open(furthest_path)
            new_img = PILImage.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
            new_img.paste(im1, (0, 0))
            new_img.paste(im2, (im1.width, 0))
            new_img.save(mixed_path)
            new_paths.append(mixed_path)
        else:
            #print(f"Warning: Furthest image not found for {impath}, keeping original.")
            new_paths.append(impath)  # keep original
    return new_paths

# Utils
def idx2time(idx, tau_sample=5, fps=25):
    #idx should be in the reference of the entire video, ie between 0 and self.dataset.n_frames
    if int(idx/fps//60)==0:
        return('{} | {:.2f}s'.format(idx//tau_sample, idx/fps%60))
    else:
        return('{} | {}m {:.2f}s'.format(idx//tau_sample, int(idx/fps//60), idx/fps%60))
    
    
def get_colors(labels):
    n_labels = len(np.unique(labels))
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors

    cmap = ListedColormap(colors[:n_labels])
    norm = plt.Normalize(vmin=0, vmax=n_labels - 1)
    return [cmap(norm(i)) for i in range(len(labels))]
