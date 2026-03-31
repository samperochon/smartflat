"""Utility function used for development

Sam Perochon @ sam.perochon@ens-paris-saclay.fr January, 2024.
"""

import matplotlib.pyplot as plt
from simple_colors import blue as _blue
from simple_colors import green as _green
from simple_colors import magenta as _magenta
from simple_colors import red as _red
from simple_colors import yellow as _yellow

# Utiliser HTML pour mettre en gras le texte


def red(s):
    print(_red(s))
def blue(s):
    print(_blue(s))
def green(s):
    print(_green(s))
def yellow(s):
    print(_yellow(s))
def purple(s):
    print(_magenta(s))
    
    
def fi(x=25, y=4):
    return plt.figure(figsize=(x,y))


def select(data,feature, value, unique_col=None):
    """Example : select(data, 'ASD+', 1, unique_col='participant_id')
                 select(data, 'clinical_sex', 'Male')
    """
    if unique_col is not None:
        data_unique=data.drop_duplicates(subset = unique_col, keep='first')
        selected_data = data_unique[data_unique[feature]==value]
    else:
        selected_data = data[data[feature]==value]
    return(selected_data)


def filter_progress_cols(df, progress_cols):
    
        
    # Filter the DataFrame to include only columns in columns_clean
    columns_to_keep = [col for col in progress_cols if col in df.columns]
    df = df[columns_to_keep]

    # Print missing columns, if any
    missing_columns = list(set(progress_cols) - set(columns_to_keep))
    if len(missing_columns) > 0:
        pass#print('Missing columns: {}'.format(missing_columns))
        
    additional_columns = list(set(columns_to_keep) - set(progress_cols))
    if len(additional_columns) > 0:
        print('New columns: {}'.format(additional_columns))
        
    return df