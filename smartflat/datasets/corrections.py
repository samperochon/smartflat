"""Manual metadata corrections for known data quality issues.

Contains hardcoded fixes for specific participant/video combinations where
raw metadata is incorrect (wrong modality labels, non-standard video names,
incorrect n_videos counts). These corrections were identified through visual
inspection of the SDS2 dataset recordings.
"""


def apply_manual_fixes(metadata):
    """Apply known corrections to the metadata DataFrame.

    Fixes three categories of issues:
    - Video name corrections (non-standard filenames)
    - Modality corrections (GoPro camera swaps)
    - n_videos count corrections (collation mismatches)

    Parameters
    ----------
    metadata : pd.DataFrame
        Video metadata with columns: participant_id, modality, video_name, n_videos.

    Returns
    -------
    pd.DataFrame
        Corrected metadata (modified in-place and returned).
    """

    # --- Video name corrections ---
    metadata.loc[
        (metadata["participant_id"] == "G83_P70_STOEri_13102021")
        & (metadata["modality"] == "GoPro1")
        & (metadata["video_name"] == "STOEri_13102021_G_1"),
        "video_name",
    ] = "GOPR0000"
    metadata.loc[
        (metadata["participant_id"] == "G83_P70_STOEri_13102021")
        & (metadata["modality"] == "GoPro1")
        & (metadata["video_name"] == "STOEri_13102021_G_2"),
        "video_name",
    ] = "GOPR0001"

    metadata.loc[
        (metadata["participant_id"] == "G83_P70_STOEri_13102021")
        & (metadata["modality"] == "GoPro3")
        & (metadata["video_name"] == "STOEric_13102021_G_1"),
        "video_name",
    ] = "GOPR0000"
    metadata.loc[
        (metadata["participant_id"] == "G83_P70_STOEri_13102021")
        & (metadata["modality"] == "GoPro3")
        & (metadata["video_name"] == "STOEric_13102021_G_2"),
        "video_name",
    ] = "GOPR0001"

    metadata.loc[
        (metadata["participant_id"] == "G16_P7_BERBea_03052017")
        & (metadata["video_name"] == "GP0108541"),
        "video_name",
    ] = "GP010854"

    # --- Modality corrections (GoPro camera swaps) ---
    metadata.loc[
        (metadata["participant_id"] == "G5_C3_FAKAzi_03032017")
        & (metadata["video_name"].isin(["GP011367", "GOPR1367"])),
        "modality",
    ] = "GoPro1"
    metadata.loc[
        (metadata["participant_id"] == "G5_C3_FAKAzi_03032017")
        & (metadata["video_name"].isin(["GP010223", "GOPR0223"])),
        "modality",
    ] = "GoPro2"
    metadata.loc[
        (metadata["participant_id"] == "G5_C3_FAKAzi_03032017")
        & (metadata["video_name"].isin(["GOPR0204", "GP010204"])),
        "modality",
    ] = "GoPro3"

    metadata.loc[
        (metadata["participant_id"] == "G16_P7_BERBea_03052017")
        & (metadata["video_name"].isin(["GOPR1388", "GP021388"])),
        "modality",
    ] = "GoPro1"
    metadata.loc[
        (metadata["participant_id"] == "G16_P7_BERBea_03052017")
        & (metadata["video_name"].isin(["GP020238", "GP010238", "GOPR0238"])),
        "modality",
    ] = "GoPro2"
    metadata.loc[
        (metadata["participant_id"] == "G16_P7_BERBea_03052017")
        & (metadata["video_name"].isin(["GOPR0854", "GP010855", "GP010854"])),
        "modality",
    ] = "GoPro3"

    metadata.loc[
        (metadata["participant_id"] == "G1_C1_BARMar_22022017")
        & (metadata["video_name"].isin(["GP010128", "GOPR0128"])),
        "modality",
    ] = "GoPro1"
    metadata.loc[
        (metadata["participant_id"] == "G1_C1_BARMar_22022017")
        & (
            metadata["video_name"].isin(
                ["GP020067", "GOPR0067", "GP010067", "GP030067"]
            )
        ),
        "modality",
    ] = "GoPro2"
    metadata.loc[
        (metadata["participant_id"] == "G1_C1_BARMar_22022017")
        & (metadata["video_name"].isin(["GOPR0166", "GP010166"])),
        "modality",
    ] = "GoPro3"

    metadata.loc[
        (metadata["participant_id"] == "G20_C11_BROJos_04072017")
        & (metadata["video_name"].isin(["GP011409", "GOPR1409"])),
        "modality",
    ] = "GoPro1"
    metadata.loc[
        (metadata["participant_id"] == "G20_C11_BROJos_04072017")
        & (metadata["video_name"].isin(["GOPR0305", "GP010305"])),
        "modality",
    ] = "GoPro2"
    metadata.loc[
        (metadata["participant_id"] == "G20_C11_BROJos_04072017")
        & (metadata["video_name"].isin(["GOPR0870", "GP010870"])),
        "modality",
    ] = "GoPro3"

    metadata.loc[
        (metadata["participant_id"] == "G8_C6_GANJea_09032017")
        & (metadata["video_name"].isin(["GOPR1370", "GP011370"])),
        "modality",
    ] = "GoPro1"
    metadata.loc[
        (metadata["participant_id"] == "G8_C6_GANJea_09032017")
        & (metadata["video_name"].isin(["GP010226", "GOPR0226"])),
        "modality",
    ] = "GoPro2"
    metadata.loc[
        (metadata["participant_id"] == "G8_C6_GANJea_09032017")
        & (metadata["video_name"].isin(["GOPR0207", "GP010207"])),
        "modality",
    ] = "GoPro3"

    metadata.loc[
        (metadata["participant_id"] == "G9_C7_MATEli_14032017")
        & (metadata["video_name"].isin(["GOPR0208", "GP010208"])),
        "modality",
    ] = "GoPro1"
    metadata.loc[
        (metadata["participant_id"] == "G9_C7_MATEli_14032017")
        & (metadata["video_name"].isin(["GOPR0227", "GP010227"])),
        "modality",
    ] = "GoPro2"
    metadata.loc[
        (metadata["participant_id"] == "G9_C7_MATEli_14032017")
        & (metadata["video_name"].isin(["GP011371", "GOPR1371"])),
        "modality",
    ] = "GoPro3"

    metadata.loc[
        (metadata["participant_id"] == "G2_P1_LEBAla_23022017")
        & (metadata["video_name"].isin(["GOPR0098", "GOPR0159"])),
        "modality",
    ] = "GoPro1"

    metadata.loc[
        (metadata["participant_id"] == "G11_P4_SAUJea_21032017")
        & (metadata["video_name"].isin(["GOPR0210", "GOPR0211", "GP010211"])),
        "modality",
    ] = "GoPro1"
    metadata.loc[
        (metadata["participant_id"] == "G11_P4_SAUJea_21032017")
        & (
            metadata["video_name"].isin(
                ["GOPR0230", "GOPR0231", "GP010231", "GP020231"]
            )
        ),
        "modality",
    ] = "GoPro2"
    metadata.loc[
        (metadata["participant_id"] == "G11_P4_SAUJea_21032017")
        & (metadata["video_name"].isin(["GOPR1374", "GOPR1375", "GP011375"])),
        "modality",
    ] = "GoPro3"

    metadata.loc[
        (metadata["participant_id"] == "G3_C2_FORCla_27022017")
        & (metadata["video_name"].isin(["GP010168", "GOPR0168"])),
        "modality",
    ] = "GoPro1"
    metadata.loc[
        (metadata["participant_id"] == "L12_P6_BRUSyl_19012018")
        & (metadata["video_name"] == "lego"),
        "modality",
    ] = "Tobii"

    # --- n_videos count corrections ---
    metadata.loc[
        (metadata["participant_id"] == "G64_P51_GEOTip_14032019")
        & (metadata["modality"] == "GoPro2"),
        "n_videos",
    ] = 3
    metadata.loc[
        (metadata["participant_id"] == "G93_P79_AMEAmo_25052022")
        & (metadata["modality"] == "GoPro2"),
        "n_videos",
    ] = 2

    # Fix unsorted videos at Percy
    metadata.loc[
        (metadata['participant_id'] == 'G110_P94_ILIMil_17052023') & (metadata['modality'] == 'Tobii'),
        'n_videos',
    ] = 1
    metadata.loc[
        (metadata['participant_id'] == 'G101_C40_MIZCel_02122022') & (metadata['modality'] == 'Tobii'),
        'n_videos',
    ] = 1
    metadata.loc[
        (metadata['participant_id'] == 'G102_P87_AUXCyr_09122022') & (metadata['modality'] == 'Tobii'),
        'n_videos',
    ] = 1
    metadata.loc[
        (metadata['participant_id'] == 'G111_P95_AMEAmo_24052023') & (metadata['modality'] == 'Tobii'),
        'n_videos',
    ] = 1
    metadata.loc[
        (metadata['participant_id'] == 'G141_P117_BAUVin_01122023') & (metadata['modality'] == 'Tobii'),
        'n_videos',
    ] = 1
    metadata.loc[
        (metadata['participant_id'] == 'G25_P14_BRUSyl_18012018') & (metadata['modality'] == 'Tobii'),
        'n_videos',
    ] = 2
    metadata.loc[
        (metadata['participant_id'] == 'G100_P86_BAUVin_25112022') & (metadata['modality'] == 'GoPro2'),
        'n_videos',
    ] = 6
    metadata.loc[
        (metadata['participant_id'] == 'G110_P94_ILIMil_17052023') & (metadata['modality'] == 'GoPro2'),
        'n_videos',
    ] = 4

    metadata.loc[metadata['video_name'] == 'merged_video', 'n_videos'] = 1

    return metadata
