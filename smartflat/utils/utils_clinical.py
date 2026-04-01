"""Clinical data utilities for the SDS2 study (Ch. 6).

Provides functions for assigning diagnosis groups to participants
in the SDS2 cohort (N=122: 26 controls, 59 TBI, 37 RIL).

The primary grouping is binary:
- 0 = Control (healthy participants, ``bras='C'``)
- 1 = Patient (TBI or RIL, ``bras='P'``)

Finer-grained pathology labels (HEALTHY, TBI, RIL) are stored in
the ``pathologie`` column of the metadata, populated by
``datasets.utils.append_clinical_data()``.
"""

import numpy as np


def diagnosis_logic(row):
    """Determine the binary diagnosis group for a participant.

    Checks clinical data fields in priority order:
    1. ``row['bras']``: 'C' -> 0 (Control), 'P' -> 1 (Patient)
    2. ``row['diag_number']``: first character 'C' -> 0, 'P' -> 1

    Parameters
    ----------
    row : pd.Series
        A row from the metadata DataFrame, expected to contain
        'bras' and 'diag_number' columns.

    Returns
    -------
    int or float
        0 for control, 1 for patient, or ``np.nan`` if no
        diagnosis information is available.
    """
    if row['bras'] == 'C':
        return 0
    elif row['bras'] == 'P':
        return 1
    elif isinstance(row['diag_number'], str) and (row['diag_number'][0] == 'C'):
        return 0
    elif isinstance(row['diag_number'], str) and (row['diag_number'][0] == 'P'):
        return 1
    else:
        print(
            f'/!\\ No clinical data found for: {row.participant_id}'
            f' - {row.modality} - trigram: {row.trigram}'
        )
        return np.nan
