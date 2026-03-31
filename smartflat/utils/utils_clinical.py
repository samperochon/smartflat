"""Utility functions related to the clinical data."""
import numpy as np


def diagnosis_logic(row):
    '''Returns the diagnosis based on the row's data.
    
    The diagnosis is retrieved using in priority:
    (1) the clinical data files (built in the demo_clinical_data notebok) if available, 
    (2) the diagnosis number retrieve from the participant_folder if available,
    
    NaN is returned if no diagnosis is found.
    '''
    
    if row['bras'] == 'C':
        return 0
    elif row['bras'] == 'P':
        return 1
    elif isinstance(row['diag_number'], str) and (row['diag_number'][0] == 'C'):
        return 0
    elif isinstance(row['diag_number'], str) and (row['diag_number'][0] == 'P'):
        return 1
    
    else:
        print(f'/!\ No clinical data found for: {row.participant_id} - {row.modality} - trigram: {row.trigram}')
        return np.nan
    

# Sippett code for the clinical data missing values (untidy...)
# clinical_data_path = os.path.join(get_data_root(), 'dataframes', 'clinical', 'merged-clinical-data-mupt.csv')
    
# cdf = pd.read_csv(clinical_data_path)
# cdf[cdf.trigram.apply(lambda x: 'lamthe' in x)]


# df = dset.metadata


# df_unknown_diag = df[(~df['group'].isin([0, 1]) |  (df['pathologie'].isna()) | (df['pathologie'] == 'OTH'))][['participant_id', 'group', 'pathologie', 'MoCA', 'ISDC']].drop_duplicates('participant_id')


# mapping_participant_id_fix_reversed = {v:k for k, v in mapping_participant_id_fix.items()}
# df_unknown_diag['autre_participant_id'] = df_unknown_diag.participant_id.map(mapping_participant_id_fix_reversed)
# df_unknown_diag['autre_participant_id']

# from smartflat.utils.utils_io import fetch_has_gaze, get_data_root, load_df, save_df


# from smartflat.utils.utils_io import parse_participant_id
# df_unknown_diag[["task_number", "diag_number", "trigram", "date_folder"]] = df_unknown_diag["participant_id"].apply(parse_participant_id)



# df_unknown_diag[['participant_id', 'autre_participant_id', 'group', 'pathologie', 'MoCA', 'ISDC']].sort_values(['pathologie']).to_csv('/Volumes/Smartflat/data-gold-final/dataframes/annotations/participants_sans_pathologie_ou_autre.csv', index=False)



# df_unknown_diag[df_unknown_diag['trigram'].apply(lambda x: x in cdf.trigram.to_list())]
