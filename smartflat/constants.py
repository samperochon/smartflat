import os

# qualification_mapping = {'K_50':{'I_N': [3, 17, 31, 44, 45, 49],
                                   
#                                    # Associated with prescribed-tasks as belonging to the dysexecutive syndroms test \in T(ask)
#                                    'I_T': [0, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14, 15, 18, 19, 20,  21, 22, 23, 24, 25, 28, 29, 30, 34, 36, 37, 38, 39, 40, 41, 42, 46, 47, 48],
#                                    # Associated with exogeneous tasks or events \in L(ife)
#                                    'I_E': [3, 16, 26, 27, 32, 43, ],
#                                    },
#                          }

# qualification_mapping['K_50'] = {'Noise': [3, 17, 31, 44, 45, 49],
                                   
#                                    # Associated with prescribed-tasks as belonging to the dysexecutive syndroms test \in T(ask)
#                                    'task': [0, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14, 15, 18, 19, 20,  21, 22, 23, 24, 25, 28, 29, 30, 34, 36, 37, 38, 39, 40, 41, 42, 46, 47, 48],
#                                    # Associated with exogeneous tasks or events \in L(ife)
#                                    'exo': [3, 16, 26, 27, 32, 43, ],
#                                    }

# Check list second reading: [14, 39] Do they remain in subsequent clustering ? Task related by default 
MEDIAN_SIGMA_RBF_N168 = 53.633

AVAILABLE_ROUND_NUMBERS = {'samperochon': [1, 2, 3, 4, 5, 6, 7, 8],
                        'theoperochon': [1, 2, 3, 4, 5, 6, 7, 8],
                        'fusionperochon': [8],}
    
progress_cols = [ 'identifier',
                    'task_name',
                    'trigram',
                    'participant_id',
                    'modality',
                    'folder_modality', 
                    'video_name',
                    
                    
                    #'task_number',
                    #'task_number_int',

                    # 'diag_number',
                    # 'diagnosis_group',
                    'group', # Used in video block dataset returns 
                    # 'groupe_bis',
                    'pathologie',
                    # 'ISDC',
                    # 'MoCA',

                    
                    'duration',
                    'fps',
                    'n_frames',
                    #'date',
                    #'date_folder',
                    'size',
                    #'processed',
                    
                    'num_frames',
                    'n_idx_embedding',
                    'n_embedding_labels',
                    
                    'has_video',
                    'has_light_video',
                    'has_annotation',
                    'has_hyperparams',
                    #'has_collate_txt',

                    
                    'n_videos',
                    'video_path',
                    #'folder_path',
                    #'registration_file',
                    'audio_modality',
                    
                    'annotation',
                    #'annotation_software',

                    'flag',
                    #'flag_video_representation',
                    #'flag_speech_recognition',
                    #'flag_speech_representation',
                    #'flag_hand_landmarks',
                    #'flag_skeleton_landmarks',


                    'video_representation_path',
                    # 'speech_recognition_path',
                    # 'speech_representation_path',
                    # 'hand_landmarks_path',
                    # 'skeleton_landmarks_path',

                    # 'video_representation_computed',
                    # 'speech_recognition_computed',
                    # 'speech_representation_computed',
                    # 'hand_landmarks_computed',
                    # 'skeleton_landmarks_computed',
                    
                    #'percy_metadata_merge',
                    #'video_metadata_merge',
                    #'hyperparameter_metadata_merge',

                    # From the consolidation...
                    # 'flag_collate_video',
                    #  'modality_id',
                    #  'video_id_list',
                    #  'video_name_list',
                    #  'dates_list',
                    #  'fps_percy',
                    #  'n_fps',
                    #  'size_list',
                    #  'size_percy',
                    #  'n_frames_list',
                    #  'n_frames_percy',
                    #  'duration_list',
                    #  'duration_percy',
                    #  'mod_identifier',
                    #  'order',
                    # 'n_identifiers_per_mod',
                    # 'has_merged_video',
                    # 'is_partition',
                    # 'is_consolidated',
                    

                    


                    # From the visual inspections
                    
                    # 'is_fish_eyed',
                    # 'upside_down',
                    # 'is_swapped_modality',
                    # 'true_modality',
                    # 'GP3_is_sink',
                    # 'GP2_is_wrong_buttress',
                    # 'GP2_above',
                    # 'GoPro1_is_wrong_buttress',
                    # 'is_middle_range',
                    # 'true_task_name',
                    # 'is_old_setup',
                    # 'is_old_recipe',
                    # 'annot_notes',

                    # 'video_id',
                    # 'is_checked',
                    
                    # From the experiments
                    #'cpts_0',
                    #'n_cpts_0',
                    #'K_0',
                    #'cpts_1',
                    #'n_cpts_1',
                    #'K_1',
                    'cpts',
                    'n_cpts',
                    'K',
                    'idx_embedding',
                    'embedding_labels',
                    
                    #'token_duration',
                    'delta_t',
                    'segment_length',
]



smartflat_features = [
    
    # 0) Descriptive columns
    'modality',  'cpts', 'cpts_percentiles', 'embedding_labels', 'segments_labels', 
    
    # I) Administration and demographics related
     
    'duration', 'has_gaze', 'has_annotation', 'age', 'group', 'pathologie',
       
       
    # II) Symbolization (Temporal segmentation  and Clustering ) related
        
        # A) Global
        
        'k_hat^p', 'lambda_0^p', 'lambda_1^p', 'k_hat^c', 'lambda_0^c', 'lambda_1^c', 'percent_embed_changes', 'sum_cpts_withdrawn',
        
        'sample_entropy_x', 'sample_entropy_s', # Other entropy measures, distance to barycenter ? QC assessment from registration quality ? 
        
          'n_segments', 'symbols_freq',
        
        # B) Per segment
        'segments_length', 'mean_cluster_dist',  'n_segmented',
        
        # C) Per embedding
        'clustet_dist',     
    
    # III) Occulometric and gaze related features
        
        # A) Global 
        
            # i) Gaze related
            'n_saccade_tot', 'n_fixation_tot', 'fixation_duration_tot_mean',
            'fixation_duration_tot_std', 'saccade_duration_tot_mean',
            'saccade_duration_tot_std', 'saccade_frequency_tot',
            'fixation_frequency_tot', 'fixation_x_tot_mean',
            'fixation_x_tot_std', 'fixation_y_tot_mean', 'fixation_y_tot_std',
            'gaze_2D_path_length_mean_tot', 'gaze_2D_path_length_std_tot',
            'gaze_3D_path_length_mean_tot', 'gaze_3D_path_length_std_tot',
            
            
            # ii) Kinematics related
            'acceleration_norm_mean_tot', 'acceleration_norm_std_tot',
            'gyro_norm_mean_tot', 'gyro_norm_std_tot',
                
                # iii) Occulometric related
            'pupil_L_diameter_mean_tot', 'pupil_L_diameter_std_tot',
            'pupil_R_diameter_mean_tot', 'pupil_R_diameter_std_tot',
            
            # iv) Validity related
            'valid_R_prop_tot', 'valid_L_prop_tot',
            
        # B) Per segment
            
            # i) Gaze related

            'n_saccades', 'n_fixation',  'saccade_frequency', 'fixation_frequency',
            'saccade_duration_mean', 'saccade_duration_std',
            'fixation_duration_mean', 'fixation_duration_std',
            'fixation_x_mean', 'fixation_x_std',
            'fixation_y_mean', 'fixation_y_std',

            'gaze_2D_path_length_mean', 'gaze_2D_path_length_std',
            'gaze_3D_path_length_mean', 'gaze_3D_path_length_std',
            'gaze_2D_norm_mean', 'gaze_2D_norm_std',
            'gaze_3D_norm_mean', 'gaze_3D_norm_std',
            
            # ii) Kinematics related

            'acceleration_norm_mean', 'acceleration_norm_std',
            'gyro_norm_mean', 'gyro_norm_std', 
            
            # iii) Occulometric related
            'pupil_L_diameter_mean', 'pupil_L_diameter_std', 
            'pupil_R_diameter_mean', 'pupil_R_diameter_std', 
            
            
            # iv) Validity related
            'valid_L_prop', 'valid_R_prop',
        
]


LOCAL_MACHINE_NAMES = [ 
        "Mac.lan",
        "egr-sjp49-mbp.local",
        'device-58.home.dhe.duke.edu',
        'device-57.home.dhe.duke.edu',
        'pclnrs119.biomedicale.univ-paris5.fr.dhe.duke.edu', 
        "device-3026.home",
        'egr-sjp49-mbp-1.home',
        'egr-sjp49-mbp-1.home.dhe.duke.edu',
        "egr-sjp49-mbp.home.dhe.duke.edu",
        "3C-06-30-12-07-86",
        'device-3026.home.dhe.duke.edu',
        "egr-sjp49-mbp.home",
        "MacOS-Sam-Perochon",
        'pclnrs103.biomedicale.univ-paris5.fr.dhe.duke.edu',
        'w-155-132.wfer.ens-paris-saclay.fr.dhe.duke.edu',
        'MacBook-Pro.local',
        'pclnrs219.biomedicale.univ-paris5.fr.dhe.duke.edu'
    ]

enabled_modalities = {'cuisine': {'flag_video_representation': ['GoPro1', 'GoPro2', 'GoPro3', 'Tobii'],
                                'flag_speech_recognition': ['GoPro1', 'GoPro2', 'GoPro3'],
                                'flag_speech_representation': ['GoPro1', 'GoPro2', 'GoPro3'],
                                'flag_hand_landmarks': ['GoPro2', 'Tobii'],
                                'flag_skeleton_landmarks': ['GoPro1'],
                                'flag_tracking_hand_landmarks': ['GoPro2', 'Tobii'],
                                },
                    
                    'lego': {'flag_video_representation': ['GoPro1', 'GoPro2', 'GoPro3', 'Tobii'],
                                'flag_speech_recognition': ['GoPro1', 'GoPro2', 'GoPro3'],
                                'flag_speech_representation': ['GoPro1', 'GoPro2', 'GoPro3'],
                                'flag_hand_landmarks': ['GoPro2', 'Tobii'],
                                'flag_tracking_hand_landmarks': ['GoPro2', 'Tobii'],
                                'flag_skeleton_landmarks': ['GoPro2']}
}




#     pids_without_videos = ['G118_P101_DAVArc_21062023', 'G121_P104_CONTit_28062023', 'G60_P47_GUIAma_24012019', 'G123_F01_SIDGab_22082023'] # Two without videos and the thirsd is wrongly associated
exluded_administrations = {
                        #'G4_P2_CHANic_02032017': 'gâteau non terminé lors de cette passation (abandon)', 
                          'GXX_P4_XXXXXX_10022017': 'corrupted video ? Ecluded', 
                           'G13_P5_HOAGui_27032017': 'à 38:42 changement de séquence... et il manque la fin de la recette sur la vidéo (programmation du four)',
                           'G160_BAIAnn_SDS2_P_Inclusion_V1_gateau': 'duplicate administration, excluded from the analysis',
                           
                           'G123_F01_SIDGab_22082023': 'Wrong order',
                           'G118_P101_DAVArc_21062023': 'Missing video', 
                           'G121_P104_CONTit_28062023': 'Missing video', 
                           'G118_P101_DAVArc_21062023': 'Missing video', 
                           
                           #'G2_P1_LEBAla_23022017': 'début difficlement identifiable perturbé par personnels nombreux et photographe dans le studio',
                           
                        #    'G19_C10_LECCar_30062017': 'il manque la fin du gâteau sur la vidéo', 
                        #    'G22_P11_ROUNaw_01092017': 'il manque la fin du gâteau sur la vidéo',
                        #    'G29_P18_TONAnn_08022018': 'il manque la fin du gâteau sur la vidéo', 
                        #    'G84_P71_BAUVin_24112021': 'il manque la fin du gâteau sur la vidéo', 
                        #    'G89_C15_LHUAna_22042022': 'il manque la fin du gâteau sur la vidéo', 
                        #    'G104_P88_LABBen_01022023': 'il manque la fin du gâteau sur la vidéo', 
                        #     'G96_P82_CABAnt_22062022': 'il manque la fin du gâteau sur la vidéo', 
                        #    'G111_P95_AMEAmo_24052023': 'il manque la fin du gâteau sur la vidéo', 
                        #    'G163_LERVer_SDS2_P_M24_V3_26062024': "il manque le début de la recette sur la vidéo", 
                            #'G138_P114_LEDel_10112023': "manque quelques secondes de la recette à la fin de la vidéo", 
                        #     'G151_P125_PROVai_27022024': "début réel à 00:02:12 car le matériel de cuisine n'était pas préparé… et il manque quelques minutes de la recette à la fin de la vidéo", 

                            #'G113_P97_MORFab_31052023': "ATTENTION : la fin de la recette est  au début de la vidéo", CORRRECTED
                            #'G115_P99_FAUJea_07062023': "ATTENTION : la fin de la recette est  au début de la vidéo",  CORRRECTED
                            #'G123_C43_SIDGab_22082023': "ATTENTION : la fin de la recette est  au début de la vidéo", CORRRECTED
                            #'G94_P80_FAUJea_08062022': "ATTENTION : la fin de la recette est au début de la vidéo + lavage de main non comptabilisé dans le temps",  CORRRECTED
                            #'G107_P91_RAYVia_03052023': "ATTENTION : la fin de la recette est au début de la vidéo + lavage de main non comptabilisé dans le temps", CORRRECTED
                            
                        #    'G27_P16_VANAnt_24012018': "gâteau commencé avant le début de l'enregistrement',
                        #    'G32_P21_PESFab_11042018': "gâteau commencé avant le début de l'enregistrement"
                        #    'G56_P43_GOBPao_20122018': "gâteau commencé avant le début de l'enregistrement", 
                        #    'G74_P61_SOUPie_21052019': "gâteau commencé avant le début de l'enregistrement",       
                        #    'G97_P83_MISABr_03102022': "gâteau commencé avant le début de l'enregistrement", 
                        #    'G145_P121_COHJul_16012024': "il manque le tout début de la recette sur la vidéo (instructions et lecture de la recette)", 
                           
                           #'G34_P23_LIEPat_16052018': 'gâteau non terminé par le patient ??!',
                           #'G35_P24_LAPGeo_05062018': 'gâteau non terminé lors de cette passation (abandon)',
                           #'G93_P79_AMEAmo_25052022': "gâteau non terminé lors de cette passation (abandon)", 
                        #    'dddddd': "", 
                           }
                           
incomplete_barycenter_administrations = {'G4_P2_CHANic_02032017': 'gâteau non terminé lors de cette passation (abandon)', 
                          'GXX_P4_XXXXXX_10022017': 'corrupted video ? Ecluded', 
                           'G13_P5_HOAGui_27032017': 'à 38:42 changement de séquence... et il manque la fin de la recette sur la vidéo (programmation du four)',
                           #'G2_P1_LEBAla_23022017': 'début difficlement identifiable perturbé par personnels nombreux et photographe dans le studio',
                           'G160_BAIAnn_SDS2_P_Inclusion_V1_gateau': 'duplicate administration, excluded from the analysis',

                           'G19_C10_LECCar_30062017': 'il manque la fin du gâteau sur la vidéo', 
                           'G22_P11_ROUNaw_01092017': 'il manque la fin du gâteau sur la vidéo',
                           'G29_P18_TONAnn_08022018': 'il manque la fin du gâteau sur la vidéo', 
                           'G84_P71_BAUVin_24112021': 'il manque la fin du gâteau sur la vidéo', 
                           'G89_C15_LHUAna_22042022': 'il manque la fin du gâteau sur la vidéo', 
                           'G104_P88_LABBen_01022023': 'il manque la fin du gâteau sur la vidéo', 
                            'G96_P82_CABAnt_22062022': 'il manque la fin du gâteau sur la vidéo', 
                           'G111_P95_AMEAmo_24052023': 'il manque la fin du gâteau sur la vidéo', 
                           'G163_LERVer_SDS2_P_M24_V3_26062024': "il manque le début de la recette sur la vidéo", 
                            #'G138_P114_LEDel_10112023': "manque quelques secondes de la recette à la fin de la vidéo", 
                            'G151_P125_PROVai_27022024': "début réel à 00:02:12 car le matériel de cuisine n'était pas préparé… et il manque quelques minutes de la recette à la fin de la vidéo", 

                           #'G113_P97_MORFab_31052023': "ATTENTION : la fin de la recette est  au début de la vidéo", 
                           #'G115_P99_FAUJea_07062023': "ATTENTION : la fin de la recette est  au début de la vidéo", 
                           #'G123_C43_SIDGab_22082023': "ATTENTION : la fin de la recette est  au début de la vidéo", 
                           # 'G94_P80_FAUJea_08062022': "ATTENTION : la fin de la recette est au début de la vidéo + lavage de main non comptabilisé dans le temps", 
                           # 'G107_P91_RAYVia_03052023': "ATTENTION : la fin de la recette est au début de la vidéo + lavage de main non comptabilisé dans le temps",

                           'G27_P16_VANAnt_24012018': "gâteau commencé avant le début de l'enregistrement",
                           'G32_P21_PESFab_11042018': "gâteau commencé avant le début de l'enregistrement",
                           'G56_P43_GOBPao_20122018': "gâteau commencé avant le début de l'enregistrement", 
                           'G74_P61_SOUPie_21052019': "gâteau commencé avant le début de l'enregistrement",       
                           'G97_P83_MISABr_03102022': "gâteau commencé avant le début de l'enregistrement", 
                           'G145_P121_COHJul_16012024': "il manque le tout début de la recette sur la vidéo (instructions et lecture de la recette)", 
                           
                           #'G34_P23_LIEPat_16052018': 'gâteau non terminé par le patient ??!',
                           #'G35_P24_LAPGeo_05062018': 'gâteau non terminé lors de cette passation (abandon)',
                           #'G93_P79_AMEAmo_25052022': "gâteau non terminé lors de cette passation (abandon)", 
                           #'dddddd': "", 
                           }
                           
incomplete_clinical_administrations ={'G4_P2_CHANic_02032017': 'gâteau non terminé lors de cette passation (abandon)', 
                          'GXX_P4_XXXXXX_10022017': 'corrupted video ? Ecluded', 
                           'G13_P5_HOAGui_27032017': 'à 38:42 changement de séquence... et il manque la fin de la recette sur la vidéo (programmation du four)',
                           #'G2_P1_LEBAla_23022017': 'début difficlement identifiable perturbé par personnels nombreux et photographe dans le studio',
                           'G160_BAIAnn_SDS2_P_Inclusion_V1_gateau': 'duplicate administration, excluded from the analysis',

                           'G19_C10_LECCar_30062017': 'il manque la fin du gâteau sur la vidéo', 
                           'G22_P11_ROUNaw_01092017': 'il manque la fin du gâteau sur la vidéo',
                           'G29_P18_TONAnn_08022018': 'il manque la fin du gâteau sur la vidéo', 
                           'G84_P71_BAUVin_24112021': 'il manque la fin du gâteau sur la vidéo', 
                           'G89_C15_LHUAna_22042022': 'il manque la fin du gâteau sur la vidéo', 
                           'G104_P88_LABBen_01022023': 'il manque la fin du gâteau sur la vidéo', 
                            'G96_P82_CABAnt_22062022': 'il manque la fin du gâteau sur la vidéo', 
                           'G111_P95_AMEAmo_24052023': 'il manque la fin du gâteau sur la vidéo', 
                           'G163_LERVer_SDS2_P_M24_V3_26062024': "il manque le début de la recette sur la vidéo", 
                            #'G138_P114_LEDel_10112023': "manque quelques secondes de la recette à la fin de la vidéo", 
                            #'G151_P125_PROVai_27022024': "début réel à 00:02:12 car le matériel de cuisine n'était pas préparé… et il manque quelques minutes de la recette à la fin de la vidéo", 

                        #    'G113_P97_MORFab_31052023': "ATTENTION : la fin de la recette est  au début de la vidéo", 
                        #    'G115_P99_FAUJea_07062023': "ATTENTION : la fin de la recette est  au début de la vidéo", 
                        #    'G123_C43_SIDGab_22082023': "ATTENTION : la fin de la recette est  au début de la vidéo", 
                        #     'G94_P80_FAUJea_08062022': "ATTENTION : la fin de la recette est au début de la vidéo + lavage de main non comptabilisé dans le temps", 
                        #     'G107_P91_RAYVia_03052023': "ATTENTION : la fin de la recette est au début de la vidéo + lavage de main non comptabilisé dans le temps",

                           'G27_P16_VANAnt_24012018': "gâteau commencé avant le début de l'enregistrement",
                           'G32_P21_PESFab_11042018': "gâteau commencé avant le début de l'enregistrement",
                           'G56_P43_GOBPao_20122018': "gâteau commencé avant le début de l'enregistrement", 
                           'G74_P61_SOUPie_21052019': "gâteau commencé avant le début de l'enregistrement",       
                           'G97_P83_MISABr_03102022': "gâteau commencé avant le début de l'enregistrement", 
                           'G145_P121_COHJul_16012024': "il manque le tout début de la recette sur la vidéo (instructions et lecture de la recette)", 
                           
                           #'G34_P23_LIEPat_16052018': 'gâteau non terminé par le patient ??!',
                           #'G35_P24_LAPGeo_05062018': 'gâteau non terminé lors de cette passation (abandon)',
                           #'G93_P79_AMEAmo_25052022': "gâteau non terminé lors de cette passation (abandon)", 
                           #'dddddd': "", 
                           }

available_tasks = ['cuisine']#  add 'lego' to also consider lego folder 
available_modality = ['Tobii']#, 'GoPro1', 'GoPro2', 'GoPro3']
available_dataset_names = ['base', 
                           'multimodal_dataset', 
                           'video_block_representation', 
                           'skeleton_landmarks', 
                           'hand_landmarks', 
                           'speech_recognition_representation']
id_cols = ['task_name', 'participant_id', 'modality']
tasks_duration_lims = {'lego': [25, 120], 'cuisine': [15, 90]} # minutes

expected_folders = ['GoPro1', 'GoPro2', 'GoPro3', 'Annotation', 'Tobii', 'Audacity']
modality_encoding = {'GoPro1': 0, 'GoPro2': 1, 'GoPro3': 2, 'Tobii': 3}
available_output_type = ['video_representation', 'speech_recognition', 'speech_representation', 'hand_landmarks', 'tracking_hand_landmarks', 'skeleton_landmarks']

ordered_cluster_types = ['task-definitive', 'exo-definitive', 'task-ambiguous', 'exo-ambiguous', 'Noise']





# TODO: keep reporting this
experiments_table = {'change_point_detection': 'change_point_detection_video_representation_Tobii_',}

# Fixing dataset

mapping_annotation_path_identifiers  = {'template_boris_path': ('example_video_name', 'modality_used', 'participant_id', 'task_name'),
                                    'BRUSYL FB.boris': ('GOPR0317', 'GoPro2', 'G25_P14_BRUSyl_18012018', 'cuisine'),
                                    'gateau2017_LECCar_MLB.boris': ('GOPR0868', 'GoPro3', 'G19_C10_LECCar_30062017', 'cuisine'),
                                    'gateau_LECCar_2017_FB.boris':  ('GOPR0868', 'GoPro3', 'G19_C10_LECCar_30062017', 'cuisine'),
                                    'BRUSyl_gateau_2018_MLB.boris':  ('GOPR0317', 'GoPro2', 'G25_P14_BRUSyl_18012018', 'cuisine'),
                                    'COUNad_SC_gateau_17012019_FB.boris':  ('GOPR8167', 'GoPro2', 'G58_P45_COUNad_17012019', 'cuisine'),
                                    
                                    'VANBru_SDS2_M12_Gateau_21042023.boris': ('GOPR8167', 'GoPro1', 'G106_P90_VANBru_21042023', 'cuisine'),
                                    'SAUJea_SC_gateau_FB.boris': ('merged_video', 'GoPro2', 'G11_P4_SAUJea_21032017', 'cuisine'), # Retrieved from summing the duration in the boris file and matching withthe closest video (<.1s error, should be the one!)
                                    'BROJos_HC_gateau_04072017_FB.boris': ('merged_video', 'GoPro2', 'G20_C11_BROJos_04072017', 'cuisine'), # Retrieved from summing the duration in the boris file and matching withthe closest video (<.1s error, and the 1/3 with the correct 60 fps)
                                    'GARMor_SC_Gateau_07072017_FB.boris': ('merged_video', 'GoPro2', 'G21_P10_GARMor_07072017', 'cuisine'),
                                    'ROUNaw_SC_gateau_FB.boris': ('merged_video', 'GoPro2', 'G22_P11_ROUNaw_01092017', 'cuisine'),
                                    
                                    
                                    'ROCFra_131017_SC_Gateua_FB.boris': ('GOPR8167', 'GoPro2', 'G23_P12_ROCFra_13102017', 'cuisine'), # This one is unsure 
                                    'ROCFra_13102017_SC_GATEAU_FB.boris': ('GOPR8167', 'GoPro2', 'G23_P12_ROCFra_13102017', 'cuisine'), # This one is unsure 
                                    'DUMJea_GATEAU_MLB.boris': ('GOPR8164', 'GoPro2', 'G55_P42_DUMJea_07122018', 'cuisine'),
                                    'DUMJea gateau FB.boris': ('GOPR8164', 'GoPro2', 'G55_P42_DUMJea_07122018', 'cuisine'),
                                    'gateau_GUIAma_2019_FB.boris': ('GOPR8169', 'GoPro2', 'G60_P47_GUIAma_24012019', 'cuisine'),
                                    
                                    
                                    'gateau_2019_ GEOTip_FB.boris': ('GOPR8176', 'GoPro2', 'G64_P51_GEOTip_14032019', 'cuisine'),
                                    'BROMAV FB.boris': ('GOPR0063', 'GoPro2', 'G65_P52_BROMav_20032019', 'cuisine'),
                                    'Gateau_VITSte_2019_FB.boris': ('GOPR0025', 'GoPro1', 'G72_P24_VITSte_14052019', 'cuisine'),
                                    
                                    'AUXANT GATEAU_MLB.boris': ('GOPR8186', 'GoPro2', 'G77_P64_AUXCyr_16072019', 'cuisine'),
                                    'AUXCyrGateau FB.boris': ('GOPR8186', 'GoPro2', 'G77_P64_AUXCyr_16072019', 'cuisine'),
                                    'gateau_JEGTon_2019_FB.boris': ('GOPR8188', 'GoPro2', 'G79_P66_JEGTon_24072019', 'cuisine'),
                                    'JEGTon_SC_gateau_24072019_FB.boris': ('GOPR8188', 'GoPro2', 'G79_P66_JEGTon_24072019', 'cuisine'),
                                    'gateau_GANJea_HC_2017_FB.boris': ('GOPR0207', 'GoPro3', 'G8_C6_GANJea_09032017', 'cuisine'), # CLosest duration to GP3...
                                    
                                    'L12_P6_BRUSyl_19012018_FB.boris': ('GOPR0884', 'GoPro3', 'L12_P6_BRUSyl_19012018', 'lego'),
                                    'SCAIsa_analyse_Mona_v2.boris': ('GOPR8194', 'GoPro2', 'L105_P85_SCAIsa_12112020', 'lego'),
                                    'SCAIsa_analyse_Mona.boris': ('GOPR8194', 'GoPro2', 'L105_P85_SCAIsa_12112020', 'lego'),
                                    'BRINic_Mona_15122020.boris': ('GOPR8210', 'GoPro2', 'L108_P86_BRINic_15122020', 'lego'),
                                    'Mona_BRINic_2020.boris': ('GOPR8210', 'GoPro2', 'L108_P86_BRINic_15122020', 'lego'),
                                    'LEBEmm_lego_2020_FB.boris': ('GOPR8208', 'GoPro2', 'L106_C21_LEBEmm_25112020', 'lego'),
                                    'JEGTon FB.boris': ('GOPR0079', 'GoPro3', 'L91_P78_JEGTon_23072019', 'lego'),
                                    'L74_P62_BROMav_FB.boris': ('GOPR0060', 'GoPro2', 'L74_P62_BROMav_18032019', 'lego'),
                                    'LIEPat FB.boris': ('GOPR0010', 'GoPro3', 'L25_P17_LIEPat_15052018', 'lego'),
                                    
                                    
                                    'GANJea_lego2017_FB.boris': ('GOPR0288', 'GoPro2', 'L5_C2_GANJea_31052017', 'lego'),
                                    
                                    'DUMJea FB.boris': ('GOPR0040', 'GoPro3', 'L52_P41_DUMJea_06122018', 'lego'),
                                    
                                    'CHABGwe_Lego1_FB.boris': ('GOPR8228', 'GoPro2', 'L125_C37_CHAGwe_11022021', 'lego'),
                                    
                                    
                                    
                                    'Hax_Gui_analyse_Mona.boris': ('GOPR8230', 'GoPro2', 'L127_P89_HAXGui_08032021', 'lego'),
                                    'HAXGui_FB.boris': ('GOPR8230', 'GoPro2', 'L127_P89_HAXGui_08032021', 'lego'),
                                    'LECCar_2018_lego_fb.boris': ('GOPR0027', 'GoPro3', 'L38_C9_LECCar_04092018', 'lego'),
                                    'MIZCel FB.boris': ('GOPR0013', 'GoPro3', 'L122_C34_MIZCel_09022021', 'lego'),
                                    'ZIAFah.boris': ('GOPR0028', 'GoPro3', 'L39_P30_ZIAFah_04092018', 'lego'),
                                    'VITSte FB.boris': ('GOPR0071', 'GoPro2', 'L83_P71_VITSte_15052019', 'lego'),
                                    'L89_P77_AUXCyr_17072019_FB.boris': ('GOPR0078', 'GoPro2', 'L89_P77_AUXCyr_17072019', 'lego'),
                                    'GAGFra_lego1_FB.boris': ('GOPR0089', 'GoPro3', 'L98_C15_GAGFra_03122019', 'lego'),
                                    
                                    
                                    
                                    'GEOTip FB.boris': ('GOPR0058', 'GoPro3', 'L73_P61_GEOTip_14032019', 'lego'),
                                    '14_OSFG_BRUSyl.boris': ('4Bz5_05m4Uh3buTcbwvTDQ==', 'Tobii', 'G25_P14_BRUSyl_18012018', 'cuisine'),
                                    '31_OSFG_LOLJiw.boris': ('y824oi7vdi-0RMuB-9bv6Q==', 'Tobii', 'G44_P31_LOLJiw_11072018', 'cuisine'),
                                    '17_OSFG_JEMBou.boris': ('xrtmCPR3vtPBRszhjemdzA==', 'Tobii', 'G28_P17_JEMBou_07022018', 'lego'),
                                    
                                    '023_ILIMil_SDS2_M12_Y2_lego_2024_FB.boris':  ('GP013534', 'GoPro2', 'L157_P116_ILIMil_17052023', 'lego'),
                                    #'DASGui_SC_gateau_16022018_FB.boris': ('GOPR0225', 'GoPro2', 'G30_P19_DASGui_16022018', 'cuisine'), # modality 
                                    '023_ILIMil_P_SDS2_M12_V2_22052024_gateau.boris':  ('GOPR0063 ', 'GoPro1', 'G110_P94_ILIMil_17052023', 'cuisine'),
                                    
                                    
                                    
                                    'LEBAla_SC_gateau_26022019_FB.boris':  ('GOPR8173', 'GoPro2', 'G62_P49_LEBAla_26022019', 'cuisine'),
                                    'JEMBou_SC_gateau_2018_FB.boris':  ('GOPR1427', 'GoPro1', 'G28_P17_JEMBou_07022018', 'cuisine'),
                                    'LOLJiw_SC_gateau_11072018_FB.boris':  ('GOPR0052', 'GoPro3', 'G44_P31_LOLJiw_11072018', 'cuisine'),
                                   
                                   
                                    'GUYAma_SC_Gateau_24012019_FB.boris':  ('GOPR8169', 'GoPro2', 'G60_P47_GUIAma_24012019', 'cuisine'),
                                    
                                    'LEBAla_SC_gateau_26022019_FB.boris':  ('GOPR8173', 'GoPro2', 'G62_P49_LEBAla_26022019', 'cuisine'),
                                    'MARNic_SC_gateau_070319_FB.boris':  ('GOPR8174', 'GoPro2', 'G63_P50_MARNic_07032019', 'cuisine'),
                                    'GEOTip_SC_gateau_14032019_FB.boris':  ('GOPR8176', 'GoPro2', 'G64_P51_GEOTip_14032019', 'cuisine'),
                                    'JOLXav_SC_gateau_2019_FB.boris':  ('GOPR8184', 'GoPro2', 'G76_P63_JOLXav_27062019', 'cuisine'),
                                    'GAUSte_SC_Gateau_2019_FB.boris':  ('GOPR0085', 'GoPro2', 'G81_P68_GAUSte_08082019', 'cuisine'),
                                    
                                    
                                    'LONCat_SC_gateau_FB.boris':  ('GOPR0031', 'GoPro3', 'G82_P69_LONCat_22092021', 'cuisine'),
                                    'JEAPie_SC_Gateau_20022019_FB.boris':  ('GOPR8172', 'GoPro2', 'G61_P48_JEAPie_20022019', 'cuisine'),
                                    'GUIAma_SC_Gateau_24012019_FB.boris':  ('GOPR8169', 'GoPro2', 'G60_P47_GUIAma_24012019', 'cuisine'),
                                    'MAINoe_SC_gateau_110718_FB.boris':  ('GOPR8174', 'GoPro2', 'G43_P30_MAINoe_11072018', 'cuisine'),
                                    '020LB_SDS2_LABben_08032024_M12_V2.boris':  ('GOPR3529', 'GoPro2', 'G152_P126_LABBen_08032024', 'cuisine'),
                                    'VIDAT AUXCYR FB gateau 2019.json':  ('dBXdSaVffQbk5Vo7CHc9pg==', 'Tobii', 'G77_P64_AUXCyr_16072019', 'cuisine'),
                                    'DUMJea2018.json': ('1_RpaJtIErlnLURub1Diug==', 'Tobii', 'G55_P42_DUMJea_07122018', 'cuisine'),

                                    'GOBPao_SC_Gateau_21122018_FB.boris': ('GOPR8165', 'GoPro2', 'G56_P43_GOBPao_20122018', 'cuisine'), # A verifier

                                    'LIEPat_SC_2018_Gateau_FB.boris': ('GOPR8631', 'GoPro3', 'G34_P23_LIEPat_16052018', 'cuisine'), # A verifier

                                    # 023_ILIMil_P_SDS2_M12_V2_22052024_gateau
                                    'BRUSYL.json': ('merged_video', 'Tobii', 'G25_P14_BRUSyl_18012018', 'cuisine'), 
                                    'DUMJea2018.json': ('1_RpaJtIErlnLURub1Diug==', 'Tobii', 'G55_P42_DUMJea_07122018', 'cuisine'),
                                    'BroMav.json': ('z-8XO3pyCr0f4xbLd7lx3g==', 'Tobii', 'G65_P52_BROMav_20032019', 'cuisine'),
                                    'BROMAV.json': ('z-8XO3pyCr0f4xbLd7lx3g==', 'Tobii', 'G65_P52_BROMav_20032019', 'cuisine'),
                                    'AuxCyr.json': ('dBXdSaVffQbk5Vo7CHc9pg==', 'Tobii', 'G77_P64_AUXCyr_16072019', 'cuisine'),
                                    'AUXCYR.json': ('dBXdSaVffQbk5Vo7CHc9pg==', 'Tobii', 'G77_P64_AUXCyr_16072019', 'cuisine'),

                                    'CHAAli_SDS2T0_gateau_FB.json': ('mF1zfXMG1i7MIwspFYMeAg==', 'Tobii', 'G85_P72_CHAAli_04022022', 'cuisine'),
                                    }


delim ='----------------------------------'
mapping_incorrect_modality_name  = {'G1_C1_BARMar_22022017_GoPro data': '?', 
                    'annotations': 'Annotations',
                    'Figures': '?',
 '01P7': '?', 
 '01P6': '?', 
 '02': '?', 
 '03': '?', 
 'gp 3': 'GoPro3',
 'Go Pro 1': 'GoPro1',
 '107GOPRO': '?', 
 'Go pro 2': 'GoPro2',
 'GoPro2_vrai': 'GoPro2',
 'Go pro 3': 'GoPro3',
 'G34_P23_LIEPat_16052018': '?', 
 '(tests': '?', 
 'G38_P27_LICVer_07062018': '?', 
 'G40_C13_DEGGui_08062018': '?', 
 'G36_P25_OLIFra_07062018': '?', 
 'G37_P26_MOYJea_07062018': '?', 
 'G41_P28_BOUDen_12062018': '?', 
 'G51_P38_ZIAFah_06092018': '?', 
 'G68_P55_ISAVal_28032019': '?', 
 'G72_P24_VITSte_14052019': '?', 
 'G77_C14_MISAnk_08072019': '?', 
 'L83_P71_VITSte_15052019': '?', 
 'L66_P55_JEAPie_19022019': '?', 
 'G74_P61_SOUPie_21052019': '?', 
 'L49_C11_TERLau_23112018': '?', 
 'L12_P6_BRUSyl_19012018': '?', 
 #'Tobii': 'Tobii', 
 'GoPro3 ok': 'GoPro3',
 'GoPro2 incomplet42': 'GoPro2',
 'G83_P70_STOEri_13102021 (2)': '?', 
 'G99_P85_BONLoi_29102022': '?', 
 'Nouveau dossier': '?', 
 'COUNad': '?', 
 '102GOPRO': '?', 
 'L169_P127_GOGFlo_18082023': '?', 
 'G122_P105_GOGFlo_18082023': '?', 
 'GoPro3manque début': 'GoPro3', #
 'L181_P137_MENYla_21112023': '?',
 'dossier sans titre': '?', 
  'Media': 'Tobii',
 #'GoPro3': 'GoPro3',
 #'GoPro1': 'GoPro1',
 'GOPRO3': 'GoPro3',
 'GOPRO1': 'GoPro1',
 'GOPRO2': 'GoPro2',
 'GoPro 02': 'GoPro2',
 'GoPro 03': 'GoPro3',
 'go pro 01': 'GoPro1',
 #'GoPro2': 'GoPro2',
 'go pro 02': 'GoPro2',
 'go pro 03': 'GoPro3',
 '1': '?',
 'G137_P113_GUIPie_09112023': '?',
 'tobii': 'Tobii',
 'gopro 01': 'GoPro1',
 'gopro 02': 'GoPro2',
 'gopro 03': 'GoPro3',
 '2': '?',
 'Export tobii': 'Tobii',
 'GoPro 3': 'GoPro3',
 'Go Pro 2': 'GoPro2',
 'Go PRO 1': 'GoPro1',
 'GO PRO 1': 'GoPro1',
 'go pro 3': 'GoPro3',
 'GP2': 'GoPro2',
 'GP2 - V ok': 'GoPro2',
 'GoPro 2': 'GoPro2',
 'gopro2': 'GoPro2',
 'gopro3': 'GoPro3',
 'GO PRO 3': 'GoPro3',
 'go pro 2': 'GoPro2',
 'Go pro 1': 'GoPro1',
 'audacity': 'Audacity',
 'gopro1': 'GoPro1',
 'go pro 1': 'GoPro1',
 'TOBII': 'Tobii',
 'Audacity_data': 'Audacity',
 'GoPro 01': 'GoPro1',
 'gro pro 3': 'GoPro3',
 'GoPro 1': 'GoPro1',
 'G49_P36_HERMic_28082018': '?',
 'go pro': '?',
 'GO PRO 2': 'GoPro2',
 'audacity_data': 'Audacity',
 'GoPro': '?',
 'Go Pro': '?',
 'Audacity data': 'Audacity',
 'GoPro data': '?',
 'TOBII data': 'Tobii',
 'GOPRO-1': 'GoPro1',
 'NCA_P2_02032017_Audacity': 'Audacity',
 'Go Pro_data': '?',
 'TOBII_data': 'Tobii',
 'Go Pro data': '?',
 'GoPro1 manque 3minfin': 'GoPro1',
 'GoPro3manque début': 'GoPro3',
 'Go pro': '?',
 'GP01-Lego-vide': 'GoPro1',
 'GP02-Lego': 'GoPro2',
 'GP03-Lego': 'GoPro3',
 'tobii dans dossier gateau': '?',
 'GoPro02': 'GoPro2',
 'enregistrement audacity_data': 'Audacity',
 'GoPro03': 'GoPro3',
 'LegoGP1': 'GoPro1',
 'LegoGP2': 'GoPro2',
 'LegoGP3': 'GoPro3',
 'Gopro': '?',
 'GO pro': '?',
 'gopro': '?',
 'GP1': 'GoPro1',
 'GP3': 'GoPro3',
 'L38_C9_LECCar_04092018': '?',
 'AUDACITY': 'Audacity',
 'audacity PMY mineureleg_data': 'Audacity',
 'GP 01': 'GoPro1',
 'GP 02': 'GoPro2',
 'GP 03': 'GoPro3'}


root_paths = [
             "F:\\Test SmartFlat 2019/test gateau",
              "F:\\Test SmartFlat 2019/test lego",
              "F:\\Test SmartFlat 2019/Test SmartFlat 2020/test lego",


              "X:\\ex-silo1/& Visites enregistrement 2018 Lego",
              "X:\\silo2/Test SmartFlat 2019/Visites enregistrement 2019 Gateau",
              #"X:\\silo2/Test SmartFlat 2019/test gateau",
              #"X:\\silo2/Test SmartFlat 2019/test lego",

              "X:\\silo2/Test SmartFlat 2020/test lego",
              "X:\\silo2/Test SmartFlat 2020/test gateau",

              "X:\\silo2/Test SmartFlat 2020/Test SmartFlat 2020/test lego",
              "X:\\silo2/Visites enregistrement 2017 Lego",
              "X:\\silo2/Visites enregistrement 2019 Lego",



              "F:\\Test SmartFlat 2021/test gateau",
              "F:\\Test SmartFlat 2021/test lego",

              'F:\\Test SmartFlat 2022/test gateau',
              'F:\\Test SmartFlat 2022/test lego',

              'E:\\test Smartflat 2023/Test Gateau',
              'E:\\test Smartflat 2023/Test Lego',

              'D:\\test smartflat 2024/tests gateau',
              'D:\\test smartflat 2024/tests lego',


            # '//S1NAS-6E-90-2A/silo1\\& Visites enregistrement 2017 Gateau',
            #  '//S1NAS-6E-90-2A/silo1\\& Visites enregistrement 2017 Lego',
            #  '//S1NAS-6E-90-2A/silo1\\& Visites enregistrement 2018 Gateau',
            #  '//S1NAS-6E-90-2A/silo1\\& Visites enregistrement 2018 Lego',
             
            #  '//S1NAS-6E-90-2A/silo2\\Test SmartFlat 2019/Visites enregistrement 2019 Gateau',
            #  '//S1NAS-6E-90-2A/silo2/Test SmartFlat 2020/test gateau',
            #  '//S1NAS-6E-90-2A/silo2/Test SmartFlat 2020/test lego',
            #  '//S1NAS-6E-90-2A/silo2/Visites enregistrement 2019 Lego',
             
            #  'E:\\test Smartflat 2023/Test Gateau',
            #  'E:\\test Smartflat 2023/Test Lego',
            #  #'E:\\test smartflat 2024/tests gateau',
            #  #'E:\\test smartflat 2024/tests lego',
             
            #  'F:\\Test SmartFlat 2019/test gateau',
            #  'F:\\Test SmartFlat 2019/test lego',
            #  #'F:\\Test SmartFlat 2020/test lego',
            #  'F:\\Test SmartFlat 2021/test gateau',
            #  'F:\\Test SmartFlat 2021/test lego',
            #  'F:\\Test SmartFlat 2022/test gateau',
            #  'F:\\Test SmartFlat 2022/test lego',

            #  "T:\ex-syno1_bis\& SmartFlat 2017-2021\Visites enregistrement 2018 Lego",
            #  "T:\ex-syno1_bis\& SmartFlat 2017-2021\Visites enregistrement 2019 Lego",
            #  "T:\ex-syno1_bis\& SmartFlat 2017-2021\Visites enregistrement 2019 Gateau",
            #  "T:\ex-syno1_bis\& SmartFlat 2017-2021\Visites enregistrement 2020 Lego",
            #  "T:\ex-syno1_bis\& SmartFlat 2017-2021\Visites enregistrement 2020 Gateau",
            #  "T:\ex-syno1_bis\& SmartFlat 2017-2021\Visites enregistrement 2021 Lego",
            #  "T:\ex-syno1_bis\& SmartFlat 2017-2021\Visites enregistrement 2021 Gateau",
             ]

mapping_boris_name_participant_id = {'LONCat_SC_gateau_FB': 'G82_P69_LONCat_22092021',
                                     'LERVER_gateau 2022_SDS2_T0_MLB': 'G86_P73_LERVer_30032022',
                                     'LERVER_gateau': 'G86_P73_LERVer_30032022',
                                     'SDS2_Y0_G119_P102_RICJon_23062023_FB': 'G119_P102_RICJon_23062023',
                                     'BROMAV FB': 'G65_P52_BROMav_20032019',
                                     'Test SmartFlat 2021\\test gateau\\G84_P71_BAUVin_24112021':  'G84_P71_BAUVin_24112021',
                                     'SDS2_Y1_gateau_FONWil_FB': 'G120_P103_FONWil_28062023',
                                     'FONWil_SDS2_T0_gateau_2022_MLB': 'G92_P78_FONWil_11052022',
                                     'COUNad_SC_gateau_17012019_FB': 'G58_P45_COUNad_17012019',
                                     'CHAAli_lego_SDS2_T0': 'L132_P94_CHAAli_04022022',
                                     'SDS2-Y1_gateau_004-LV_LERVer_14062023_FB': 'G117_P100_LERVer_14062023',
                                     'LOLJiw_SC_gateau_11072018_FB': 'G44_P31_LOLJiw_11072018', 
                                     'DASGui_SC_gateau_16022018_FB': 'G30_P19_DASGui_16022018',
                                     'GAUSte_SC_gateau_20082019': 'G81_P68_GAUSte_08082019',
                                     'SDS2_Y0_016_GA_gateau_FB': 'G98_P84_GIBAnt_07102022',
                                     'SDS2_Y0_012_FJ_gateau_FB': 'G94_P80_FAUJea_08062022',
                                     'VANBru_SDS2_T0_gateau_FB': 'G87_P74_VANBru_06042022',
                                     '023_IM_SDS2_T0_gateau': 'G110_P94_ILIMil_17052023',
                                     'GEOTip_SC_gateau_14032019_FB': 'G64_P51_GEOTip_14032019',
                                     'SDS2_Y1_012-FJ_gateau_FB': 'G115_P99_FAUJea_07062023',
                                     'Gateau_VITSte_2019_FB': 'G72_P24_VITSte_14052019',
                                     'CHAAli_gateau_sds2_MLB': 'G85_P72_CHAAli_04022022', # Not sure ['G85_P72_CHAAli_04022022' 'G112_P96_CHAAli_26052023'] 
                                     'SDS2_Y0_024-MF_gateau_FB_GoPro1': 'L82_P70_FREGui_15052019',
                                     'gateau_2019_ GEOTip_FB': 'G64_P51_GEOTip_14032019', # Doublon
                                     'G84_P71_BAUVin_24112021': 'G84_P71_BAUVin_24112021'
                                    }
fix_dumjea_path = '/Users/samperochon/Borelli/data/dataframes/fix_DUMJEA.csv'


mapping_participant_id_fix = {

    # A completer
    'G6_C4_PG_08032017': 'G6_C4_PxxGxx_08032017',
    'G12_C8_SN_22032017': 'G12_C8_SNxxxx_22032017', 
    'G77_P64_AUXCyr_16072049': 'G77_P64_AUXCyr_16072019',
    'G111_AMEAmou_24052023': 'G111_P95_AMEAmo_24052023',
    
    'G93_P79_AMEAmou_25052022': 'G93_P79_AMEAmo_25052022',

    'G124_BRASte_240823': 'G124_PXX_BRASte_240823', 
    'SC_BRASte_G124_240823': 'G124_PXX_BRASte_240823',
    'SC_G124_PXX_BRASte_24082023': 'G124_PXX_BRASte_240823',

    'G138_P114__LEDel_10112023': 'G138_P114_LEDel_10112023',
    'SC_G138_P114__LEDel_10112023': 'G138_P114_LEDel_10112023',
    'SC_G138_P114_LEDel_10112023': 'G138_P114_LEDel_10112023',
    'SC_G138_P114__LEDel_10112023': 'G138_P114_LEDel_10112023',
    
    'SC_G133_C48_LEBEmm_05102023': 'SC_G133_P114_LEBEmm_05102023',
    'SC_G130_C45_RICDam_29092023': 'SC_G130_P111_RICDam_29092023',
    'SC_G129_C44_DELMAr_27092023': 'SC_G129_P110_DELMAr_27092023',
    'G150_MIZCel_16022024': 'G150_CXX_MIZCel_16022024',
    'SC_G132_C47_VILJea_04102023': 'SC_G132_P113_VILJea_04102023',

    'G140_P116_MISAbra_29112023': 'G140_P116_MISAbr_29112023',
    'G149_CHAGwe_16022024': 'G149_CXX_CHAGwe_16022024', 
    'G150__MIZCel_16022024': 'G150_CXX_MIZCel_16022024',
    #'G150_CXX_MIZCel_16022024': 'G150_MIZCel_16022024',
    # 'G159_RAYVia_SDS2_M12_V2_31052024_gateau': 'G159_RAYVia_SDS2_M12_V2_31052024',
    # 'G160_BAIAnn_SDS2_P_Inclusion_V1_gateau': 'G160_BAIAnn_SDS2_P_Inclusion_V1',
    # 'G161_AMEAmo_SDS2_P_14062024_ M24_V3_gateau': 'G161_AMEAmo_SDS2_P_14062024_M24_V3',
    # 'G162_FAUJea_SDS2_P_19062024_M24_V3_gateau': 'G162_PXXX_FAUJea_19062024',
    # 
    # 'G163_LERVer_SDS2_P_26062024_M24_V3_gateau': 'G163_PXXX_LERVer_26062024',
    # 'G165_FONWil__SDS2_P_M24_V3_28062024_gateau': 'G165_PXXX_FONWil_28062024',
    # 'G166_FERVal__SDS2_C_Inclusion_V1_27062024_gateau': 'G166_CXXX_FERVal_27062024',
    # 'G167_BOUAde_SDS2_P_M24_V3_02072024_gateau': 'G167_PXXX_BOUAde_02072024',
    # 'G168_COUAnt_SDS2_P_M24_V3_05072024_gateau': 'G168_PXXX_COUAnt_05072024',

    'L140_P101_AMEAmou_25052022': 'L140_P101_AMEAmo_25052022',
    'L154_P113__RAYVia_03052023': 'L154_P113_RAYVia_03052023',
    'L158_P117_AMEAmou_24052023': 'L158_P117_AMEAmo_24052023',
    'L182_P138_MISAbra_29112023': 'L182_P138_MISAbr_29112023',
    'L191_CHAGwe_16022024': 'L191_CXX_CHAGwe_16022024',
    'L192_MIZCel_16022024': 'L192_CXX_MIZCel_16022024',

    # 'L201_RAYVia_SDS2_M12_V2_31052024_lego': 'L201_PXXX_RAYVia_31052024',
    # 'L202_BAIAnn_SDS2_P_Inclusion_V1_lego': 'L202_PXXX_BAIAnn_XXXXXXXX',
    # 'L203_AMEAmo_SDS2_P_14062024_ M24_V3_lego': 'L203_PXXX_AMEAmo_14062024',
    # 'L204_FAUJea_SDS2_P_19062024_M24_V3_lego': 'L204_PXXX_FAuJea_19062024',
    # 'L205_LERVer_SDS2_P_26062024_M24_V3_lego': 'L205_PXXX_LERVer_26062024',
    # 'L207_FONWil__SDS2_P_M24_V3_28062024_lego': 'L207_PXXX_FONWil_28062024',
    # 'L208_FERVal__SDS2_C_Inclusion_V1_27062024_lego': 'L208_CXXX_FERVal_27062024',
    # 'L209_BOUAde_SDS2_P_02072024_M24_lego': 'L209_PXXX_BOUAde_02072024',
    # 'L210_COUAnt_SDS2_P_M24_V3_05072024_lego': 'L210_PXXX_COUAnt_05072024',


    'PM_VB_25062019': 'LXX_PYY_PMxVBx_25062019',
    'LXX_PYY_PMVB_25062019': 'LXX_PYY_PMxVBx_25062019',
    
    
    
    'L144_P105_MISAbr_03102022': 'L144_P105_MISABr_03102022',



    'Lamboust Théo soins courant': 'LXX_PYY_LAMThe_15062022',
    'Val_de_Grace_P4_10022017( C-E)': 'GXX_P4_XXXXXX_10022017',

    'PMYmineure_L_23022018': 'LXX_PYY_PMY_23022018',
    'LXX_PYY_PMY_23022018': 'LXX_PYY_PMYxxx_23022018',

    'SC_BRASSte_L171_240823': 'SC_L171_PXX_240823',
    'SC-L180_P136_LEDel_10112023': 'SC_L180_P136_LEDel_10112023',
    'SC_L173_P129__GOUJul_06092023': 'SC_L173_P129_GOUJul_06092023',
    
    
    "L175_P131_ZIGLud_22092023": 'L175_P131_ZIELud_22092023', 
    "L194_P147_LABben_08032024": "L194_P147_LABBen_08032024", 
    "L155_P114_LONCat_050523": "L155_P114_LONCat_05052023",
    "L31_P23_ARNChr_0307208": "L31_P23_ARNChr_03072018", 
    "L14_P8_CARFra_23012017": "L14_P8_CARFra_23012018", 

    
    
    'G128_P109_ZIGLud_22092023':'G128_P109_ZIELud_22092023',
    'G152_P126_LABben_08032024':'G152_P126_LABBen_08032024',
    'G155_P_VANBru_SDS2_24042024_V3_M36_gateau':'G155_VANBru_SDS2_P_M24_V3_24042024',
    'G156_P_CONTit_SDS2_V2_M12_16052024_gateau':'G156_CONTit_SDS2_P_M12_V2_16052024',
    'G157_P_ILIMil_SDS2_M12_V2_22052024_gateau':'G157_ILIMil_SDS2_P_M12_V2_22052024',
    'G158_P_LONCat_SDS2_M24_V3_29052024_gateau':'G158_LONCat_SDS2_P_M24_V3_29052024',
    'G159_RAYVia_SDS2_M12_V2_31052024_gateau':'G159_RAYVia_SDS2_P_M12_V2_31052024',
    'G160_BAIAnn_SDS2_P_12062024_Inclusion_V1_gateau':'G160_BAIAnn_SDS2_P_M0_V1_12062024',
    'G160_BAIAnn_SDS2_P_Inclusion_V1_gateau': 'G160_BAIAnn_SDS2_P_M0_V1_12062024',
    'G161_AMEAmo_SDS2_P_14062024_ M24_V3_gateau':'G161_AMEAmo_SDS2_P_M24_V3_14062024',
    'G162_FAUJea_SDS2_P_19062024_M24_V3_gateau':'G162_FAUJea_SDS2_P_M24_V3_19062024',
    'G163_LERVer_SDS2_P_26062024_M24_V3_gateau':'G163_LERVer_SDS2_P_M24_V3_26062024',
    'G164_LHUAna_SDS2_C_M24_V3_27062024_gateau':'G164_LHUAna_SDS2_C_M24_V3_27062024',
    'G165_FONWil_SDS2_P_M24_V3_28062024_gateau':'G165_FONWil_SDS2_P_M24_V3_28062024',
    'G165_FONWil__SDS2_P_M24_V3_28062024_gateau':'G165_FONWil_SDS2_P_M24_V3_28062024',
    'G166_FERVal__SDS2_C_Inclusion_V1_27062024_gateau':'G166_FERVal_SDS2_C_M0_V1_27062024',
    'G167_BOUAde_SDS2_P_M24_V3_02072024_gateau':'G167_BOUAde_SDS2_P_M24_V3_02072024',
    'G168_COUAnt_SDS2_P_M24_V3_05072024_gateau':'G168_COUAnt_SDS2_P_M24_V3_05072024',
    'G169_CAPOph_SDS2_C_I_V1_10072024_gateau':'G169_CAPOph_SDS2_C_M0_V1_10072024',
    'G170_MARThi_SDS2_C_I_V1_11072024':'G170_MARThi_SDS2_C_M0_V1_11072024',
    'G171_SABNic_SDS2_P_I_V1_12072024':'G171_SABNic_SDS2_P_M0_V1_12072024',
    'G172_DUROli_SDS2_C_I_V1_12072024':'G172_DUROli_SDS2_C_M0_V1_12072024',
    'G174_LHUAna_SDS2_P_M24_V3_06082024':'G174_LHUAna_SDS2_C_M24_V3_06082024',
    'G178_PITAla_SDS2_P_I_V1_18092024':'G178_PITAla_SDS2_P_M0_V1_18092024',
    'G179_HERSev_SDS2_P_I_V1_19092024':'G179_HERSev_SDS2_P_M0_V1_19092024',



    "L197_P_VANBru_SDS2_24042024_V3_M36_lego": "L197_VANBru_SDS2_P_M24_V3_24042024",
    "L198_P_CONTit_SDS2_V2_M12_16052024_lego": "L198_CONTit_SDS2_P_M12_V2_16052024",
    "L199_P_ILIMil_SDS2_V2_M12_22052024_lego": "L199_ILIMil_SDS2_P_M12_V2_22052024",
    "L200_P_LONCat_SDS2_M24_V3_29052024_lego": "L200_LONCat_SDS2_P_M24_V3_29052024",
    "L201_RAYVia_SDS2_M12_V2_31052024_lego": "L201_RAYVia_SDS2_P_M12_V2_31052024",
    "L202_BAIAnn_SDS2_P_12062024_Inclusion_V1_lego": "L202_BAIAnn_SDS2_P_M0_V1_12062024",
    "L203_AMEAmo_SDS2_P_14062024_ M24_V3_lego": "L203_AMEAmo_SDS2_P_M24_V3_14062024",
    "L204_FAUJea_SDS2_P_19062024_M24_V3_lego": "L204_FAUJea_SDS2_P_M24_V3_19062024",
    "L205_LERVer_SDS2_P_26062024_M24_V3_lego": "L205_LERVer_SDS2_P_M24_V3_26062024",
    "L206__LHUAna_SDS2_C_M24_V3_27062024_lego": "L206_LHUAna_SDS2_C_M24_V3_27062024",
    "L206_FONWil_SDS2_P_28062024_M24_V3_lego": "L207_FONWil_SDS2_P_M24_V3_28062024",
    "L208_FERVal__SDS2_C_Inclusion_V1_27062024_lego": "L208_FERVal_SDS2_C_M0_V1_27062024",
    "L209_BOUAde_SDS2_P_02072024_M24_lego": "L209_BOUAde_SDS2_P_M24_V3_02072024",
    "L210_COUAnt_SDS2_P_M24_V3_05072024_lego": "L210_COUAnt_SDS2_P_M24_V3_05072024",
    "L211_CAPOph_SDS2_C_I_V1_10072024_lego": "L211_CAPOph_SDS2_C_M0_V1_10072024",
    "L212_MARThi_SDS2_C_I_V1_11072024":	"L212_MARThi_SDS2_C_M0_V1_11072024",
    "L213_SABNic_SDS2_P_I_V1_12072024":	"L213_SABNic_SDS2_P_M0_V1_12072024",
    "L214_DUROli_SDS2_C_I_V1_12072024":	"L214_DUROli_SDS2_C_M0_V1_12072024",
    "L216_LHUAna_SDS2_P_M24_V3_06082024":	"L216_LHUAna_SDS2_C_M24_V3_06082024",
    "L220_PITAla_SDS2_P_I_V1_18092024":	"L220_PITAla_SDS2_P_M0_V1_18092024",
    "L221_HERSev_SDS2_P_I_V1_19092024":	"L221_HERSev_SDS2_P_M0_V1_19092024",


    # 'LEBALA_G62': 'G62_P49_LEBALA_26022019',
    # 'LIEPAT2019': 'G71_P23_LIEPat_07052019',
    # 'LIEPAT': 'G34_P23_LIEPat_16052018',
    # 'AUXCYR2019': 'G77_P64_AUXCyr_16072019',
    # 'AUXCYR': 'G102_P87_AUXCYR_0912202',
    # 'GUIANA': 'G60_P47_GUIAma_24012019',
    # 'GEOTip': 'G64_P51_GEOTip_14032019',
    # 'GEOTiph': 'G64_P51_GEOTip_14032019',
    # 'COUNad': 'GXX_PXX_COUNad_XXXXXXXX',
    #'AMEAMO': 'G93_P79_AMEAmo_25052022',
    #'AMEAMO':'L140_P101_AMEAmou_25052022'
}
# Check the black list of hard-coded paths
# Unfortunately this is necessary as sthe way to map the anotations
# To the actual administration is through looking at the original video path at the PErcy computer
# Sometimes the path does not follow standardization and we have to hard-code the exceptions

hard_parsed_path = {'F:/Test Smartflat 2022/test lego/L148_C40_MIZCel_02122022/GoPro2/Nouveau dossier/GOPR8311.MP4': ('GOPR8311', 'GoPro2', 'L148_C40_MIZCel_02122022', 'lego'),
                        'E:/video 2024 ap080324inclus/G152_P126_LABben_08032024/GoPro2/GOPR3529.MP4': ('GOPR3529', 'GoPro2', 'G152_P126_LABBen_08032024', 'cuisine'),
                     'E:/video 2024 ap080324inclus/ILIMil_SDS2_M12_gateau/GoPro1/GOPR0258.MP4': ('GOPR0258', 'GoPro1', 'G110_P94_ILIMil_17052023', 'cuisine'),
                       'E:/video 2024 ap080324inclus/ILIMil_SDS2_M12_lego/GoPro2/GOPR3534.MP4':('GOPR3534', 'GoPro2', 'L157_P116_ILIMil_17052023', 'lego')
                        
                        }


video_extensions = [
    ".mp4",
    ".MP4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".mpg",
    ".mpeg"]

debug_columns = ['identifier', 'video_name', 'video_path', 'n_videos', 'processed', 
                'speech_recognition_computed', 'speech_representation_computed', 'video_representation_computed','hand_landmarks_computed','skeleton_landmarks_computed',
                'flag_speech_recognition','flag_speech_representation', 'flag_video_representation', 'flag_hand_landmarks','flag_skeleton_landmarks', 'flag_collate_video']


flag_columns = ['flag_video_representation', 'flag_speech_recognition','flag_speech_representation', 'flag_hand_landmarks','flag_skeleton_landmarks']
# TODO: do a doc to explain the following columns. Associated  with the fit_and_solves_cpts_curve.py
exp_cols = [
    "task_name",
    "participant_id",
    "modality",
    "identifier",
    "fps",
    "n_frames",
    "N",
    "normalization",
    "n_pca_components",
    "whiten",
    "duration",
    "n_frames_binned",
    "duration_labels",
    "duration_labels_counts",
    "experiment_id",  # Keep one random for tracking purposes
    "L_hat^p",
    "k_hat^p",
    "x0_hat^p",
    "L_var^p",
    "k_var^p",
    "x0_var^p",
    "lambda_0_log^p",
    "lambda_1_log^p",
    "lambda_0^p",
    "lambda_1^p",
    "L_hat^c",
    "k_hat^c",
    "x0_hat^c",
    "L_var^c",
    "k_var^c",
    "x0_var^c",
    "lambda_0_log^c",
    "lambda_1_log^c",
    "lambda_0^c",
    "lambda_1^c",
]


# TODO clean deprecated ?
######################################################
# PATHS
######################################################

#### Absolute path
# SRC_DIR = '/Users/samperochon/Borelli/work/code/temporal_segmentation'
# INPUT_DIR = "/Users/samperochon/Borelli/work/data/"
# OUTPUT_DIR = "/Users/samperochon/Borelli/work/data/outputs"

# SRC_DIR = '/home/perochon/temporal_segmentation/temporal_segmentation/'
# INPUT_DIR = "/home/perochon/temporal_segmentation/data/"
# OUTPUT_DIR = "/home/perochon/temporal_segmentation/outputs/"
# TWFINCH_PATH = '/home/perochon/temporal_segmentation/tmp/FINCH-Clustering/TW-FINCH/'
# TWFINCH_PATH = "/Users/samperochon/Borelli/work/code/temporal_segmentation/tools/FINCH-Clustering/TW-FINCH/"


#### Try to put all path in a relative fashion compared to this tools/const.py file!
# SRC_DIR = '/Users/samperochon/Borelli/algorithms/temporal_segmentation/'
# #DATA_DIR = os.path.join(SRC_DIR, 'data')
# #OUTPUT_DIR = os.path.join(SRC_DIR, 'data', 'outputs')
# # TODO remove dependency
# TWFINCH_PATH = './FINCH-Clustering/TW-FINCH/'


# # Path are provided with the assumption that the root_dir is determined within scripts, and the following files are within the /data folder of the projects
# DATASET_VIDEOS_SMARTFLAT = 'dataframes/Smartflat_dataset_recap.csv'
# ANNOTATION_NOMENCLATURE_PATH = "dataframes/tableau_annotation_cuisine_Smartflat.csv"
# CLINICAL_DATA_PATH = "dataframes/lego_cake_table_with_id.csv"
# FIX_DUMJEAN_PATH = 'dataframes/fix_DUMJEA.csv'
# CLINICAL_SCORES_PATH = 'dataframes/Scores_cliniques.csv'


# MAPPING_VIDEOS_ORDER = {'cuisine':{'GoPro1':{},
#                                    'GoPro2': {},
#                                    'GoPro3': {},
#                                    'Tobii': {'BRUSYL': {'suite gateau': 1, 'debutgateau06min08sec': 0},
#                                              'FAUJEA': {'IHNlH32uUdFq9dlkYbvl3g==': 1, 'baZskXMLJRtjL05jT9TlrQ==': 0},
#                                              'GAGFRA': {'Uyy2OYNGcH5HaBAi6bzU3w==': 1, 'ChINE4jVs3jUDuRWDJ6e2Q==': 0},
#                                              'LHUANA': {'SDOuIOymwq1NRVZRi3ZU6A==': 0, 'VWCr0i9Mj63Xyg_tN6bS5g==': -1},
#                                              'LIEPAT2019': {'BdviYVmDVI-ivUPSuquJ2w==': 0, 'CALjkOz49V3ehLQhovOb-A==': 1}
#                                              }

#                                    },
#                         'lego':{'GoPro1':{},
#                                    'GoPro2': {},
#                                    'GoPro3': {},
#                                    'Tobii': {}
#                               },
#                         'Lego':{'GoPro1':{},
#                                    'GoPro2': {},
#                                    'GoPro3': {},
#                                    'Tobii': {}
#                               }
#                         }


# rsync -ahuvz 'cheetah:/diskA/sam_data/data/experiments/change_point_detection/Tobii_lambda_*' /Users/samperochon/Borelli/data/experiments/change_point_detection/


# scp  /Volumes/Smartflat/data/dataframes/persistent_metadata/*_vid* ruche:/gpfs/workdir/perochons/data/dataframes/persistent_metadata/
#  scp cheetah:/diskA/sam_data/data/dataframes/persistent_metadata/'*video_metadata.csv' /Volumes/Smartflat/data/dataframes/persistent_metadata#  scp cheetah:/diskA/sam_data/data/dataframes/persistent_metadata/'*video_metadata.csv' /Volumes/Smartflat/data/dataframes/persistent_metadata




### GAZE DATA

            
gaze_features = {'gaze_event_data':['n_saccades', 'n_fixation',
                                    'saccade_duration_mean', 'saccade_duration_std', 
                                    'fixation_duration_mean', 'fixation_duration_std', 
                                    'saccade_frequency', 'fixation_frequency', 
                                    'fixation_x_mean', 'fixation_x_std',
                                    'fixation_y_mean', 'fixation_y_std',
                                    
                                    'n_saccade_tot', 'n_fixation_tot',
                                    'fixation_duration_tot_mean', 'fixation_duration_tot_std',
                                    'saccade_duration_tot_mean', 'saccade_duration_tot_std',
                                    'saccade_frequency_tot', 'fixation_frequency_tot',
                                    'fixation_x_tot_mean', 'fixation_x_tot_std',
                                    'fixation_y_tot_mean', 'fixation_y_tot_std'
                                    ],
                 
                    'accelerometric_data':['acceleration_norm_mean', 'acceleration_norm_std', 'acceleration_norm_mean_tot', 'acceleration_norm_std_tot'],
                   
                    'gyroscopic_data':['gyro_norm_mean', 'gyro_norm_std', 'gyro_norm_mean_tot', 'gyro_norm_std_tot'],
                    
                    'gaze_data':['gaze_2D_norm_mean', 'gaze_2D_norm_std', 'gaze_3D_norm_mean', 'gaze_3D_norm_std', 
                                 'pupil_L_diameter_mean', 'pupil_L_diameter_std', 'pupil_R_diameter_mean', 'pupil_R_diameter_std', 
                                 'valid_L_prop', 'valid_R_prop', 'valid_R_prop_tot', 'valid_L_prop_tot',
                                 
                                 'gaze_2D_path_length_mean', 'gaze_2D_path_length_std', 'gaze_3D_path_length_mean', 'gaze_3D_path_length_std', 
                                    'pupil_L_diameter_mean_tot', 'pupil_L_diameter_std_tot', 'pupil_R_diameter_mean_tot', 'pupil_R_diameter_std_tot',
                                 'gaze_2D_path_length_mean_tot', 'gaze_2D_path_length_std_tot', 'gaze_3D_path_length_mean_tot', 'gaze_3D_path_length_std_tot']}

gaze_type_columns_mapping = {'gaze_event_data': ['time', 'eye_movement_type', 'event_duration', 'eye_movement_type_index', 'fixation_point_x', 'fixation_point_y'],
                             'gaze_data': ['time', 'duration_gaze', 'gaze_2D_x', 'gaze_2D_y', 'gaze_3D_x', 'gaze_3D_y',
                                            'gaze_3D_z', 'gaze_direction_L_x', 'gaze_direction_L_y', 'gaze_direction_L_z',
                                            'gaze_direction_R_x', 'gaze_direction_R_y', 'gaze_direction_R_z',
                                            'pupil_L_x', 'pupil_L_y', 'pupil_L_z', 'pupil_R_x', 'pupil_R_y',
                                            'pupil_R_z', 'pupil_L_diameter', 'pupil_R_diameter', 'valid_L',
                                            'valid_R', 'fixation_point_x', 'fixation_point_y'
                                        ],
                             'accelerometric_data': ['time','acceleration_x', 'acceleration_y',  'acceleration_z'],
                             'gyroscopic_data': ['time', 'gyro_x', 'gyro_y', 'gyro_z']}



gaze_useful_columns = ['Recording timestamp [μs]', 'Sensor',
                       'Recording duration',
                   'Gaze point X [MCS px]', 'Gaze point Y [MCS px]',
                   'Gaze point 3D X [HUCS mm]', 'Gaze point 3D Y [HUCS mm]',
                   'Gaze point 3D Z [HUCS mm]', 'Gaze direction left X [HUCS norm]',
                   'Gaze direction left Y [HUCS norm]',
                   'Gaze direction left Z [HUCS norm]',
                   'Gaze direction right X [HUCS norm]',
                   'Gaze direction right Y [HUCS norm]',
                   'Gaze direction right Z [HUCS norm]', 'Pupil position left X [HUCS mm]',
                   'Pupil position left Y [HUCS mm]', 'Pupil position left Z [HUCS mm]',
                   'Pupil position right X [HUCS mm]', 'Pupil position right Y [HUCS mm]',
                   'Pupil position right Z [HUCS mm]', 'Pupil diameter left',
                   'Pupil diameter right [mm]', 'Validity left', 'Validity right', 'Eye movement type',
                   'Gaze event duration [ms]', 'Eye movement type index',
                   'Fixation point X [MCS px]', 'Fixation point Y [MCS px]',
                   'Gyro X [°/s]', 'Gyro Y [°/s]', 'Gyro Z [°/s]',
                   'Accelerometer X [m/s²]', 'Accelerometer Y [m/s²]',
                   'Accelerometer Z [m/s²]']

# gaze_columns_mapping = {"Recording timestamp [μs]": "time",
#                      "Sensor" : "sensor",
#                    "Gaze point X [MCS px]" : "gaze_2D_x", 
#                    "Gaze point Y [MCS px]" : "gaze_2D_y", 
#                    "Gaze point 3D X [HUCS mm]" : "gaze_3D_x", 
#                    "Gaze point 3D Y [HUCS mm]" : "gaze_3D_y", 
#                    "Gaze point 3D Z [HUCS mm]" : "gaze_3D_z", 
#                    "Gaze direction left X [HUCS norm]" : "gaze_direction_L_x", 
#                     "Gaze direction left Y [HUCS norm]" : "gaze_direction_L_y", 
#                     "Gaze direction left Z [HUCS norm]" : "gaze_direction_L_z", 
#                    "Gaze direction right X [HUCS norm]" : "gaze_direction_R_x", 
#                     "Gaze direction right Y [HUCS norm]" : "gaze_direction_R_y", 
#                     "Gaze direction right Z [HUCS norm]" : "gaze_direction_R_z", 
#                     "Pupil position left X [HUCS mm]" : "pupil_L_x", 
#                     "Pupil position left Y [HUCS mm]" : "pupil_L_y", 
#                     "Pupil position left Z [HUCS mm]" : "pupil_L_z", 
#                     "Pupil position right X [HUCS mm]" : "pupil_R_x", 
#                     "Pupil position right Y [HUCS mm]" : "pupil_R_y", 
#                     "Pupil position right Z [HUCS mm]" : "pupil_R_z", 
#                     "Pupil diameter left [mm]" : "pupil_L_diameter",
#                     "Pupil diameter right [mm]" : "pupil_R_diameter",
#                     "Validity left" : "valid_L",
#                     "Validity right" : "valid_R",
#                     "Eye movement type" : "eye_movement_type",
#                     "Gaze event duration [ms]" : "event_duration",
#                     "Eye movement type index" : "eye_movement_type_index",
#                     "Fixation point X [MCS px]" : "fixation_point_x",
#                     "Fixation point Y [MCS px]" : "fixation_point_y",
#                     "Gyro X [°/s]" : "gyro_x",
#                     "Gyro Y [°/s]" : "gyro_y",
#                     "Gyro Z [°/s]" : "gyro_z",
#                     "Accelerometer X [m/s²]" : "acceleration_x",
#                     "Accelerometer Y [m/s²]" : "acceleration_y",
#                     "Accelerometer Z [m/s²]" : "acceleration_z"}

gaze_columns_mapping = {"Recording timestamp": "time",
                     "Sensor" : "sensor",
                    "Recording duration" : "gaze_duration",
                   "Gaze point X" : "gaze_2D_x", 
                   "Gaze point Y" : "gaze_2D_y", 
                   "Gaze point 3D X" : "gaze_3D_x", 
                   "Gaze point 3D Y" : "gaze_3D_y", 
                   "Gaze point 3D Z" : "gaze_3D_z", 
                   "Gaze direction left X" : "gaze_direction_L_x", 
                    "Gaze direction left Y" : "gaze_direction_L_y", 
                    "Gaze direction left Z" : "gaze_direction_L_z", 
                   "Gaze direction right X" : "gaze_direction_R_x", 
                    "Gaze direction right Y" : "gaze_direction_R_y", 
                    "Gaze direction right Z" : "gaze_direction_R_z", 
                    "Pupil position left X" : "pupil_L_x", 
                    "Pupil position left Y" : "pupil_L_y", 
                    "Pupil position left Z" : "pupil_L_z", 
                    "Pupil position right X" : "pupil_R_x", 
                    "Pupil position right Y" : "pupil_R_y", 
                    "Pupil position right Z" : "pupil_R_z", 
                    "Pupil diameter left" : "pupil_L_diameter",
                    "Pupil diameter right" : "pupil_R_diameter",
                    "Validity left" : "valid_L",
                    "Validity right" : "valid_R",
                    "Eye movement type" : "eye_movement_type",
                    "Gaze event duration" : "event_duration",
                    "Eye movement type index" : "eye_movement_type_index",
                    "Fixation point X" : "fixation_point_x",
                    "Fixation point Y" : "fixation_point_y",
                    "Gyro X" : "gyro_x",
                    "Gyro Y" : "gyro_y",
                    "Gyro Z" : "gyro_z",
                    "Accelerometer X" : "acceleration_x",
                    "Accelerometer Y" : "acceleration_y",
                    "Accelerometer Z" : "acceleration_z"}