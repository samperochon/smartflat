from collections import defaultdict
import pandas as pd


analysis_mapping_K_space = []

analysis_mapping_K_space.append({'name': 'recipe',
                         'cluster_index': [401]})
analysis_mapping_K_space.append({'name': 'prepare mixer',
                         'cluster_index': []})
analysis_mapping_K_space.append({'name': 'talking',
                         'cluster_index': [137, 487, 352, 644]})
analysis_mapping_K_space.append({'name': 'look window',
                         'cluster_index': []})
analysis_mapping_K_space.append({'name': 'manipulate oven',
                         'cluster_index': [379, 534, 266, 129]})
analysis_mapping_K_space.append({'name': 'setting oven',
                         'cluster_index': [504, 97, 313]})
analysis_mapping_K_space.append({'name': 'exogeneous',
                         'cluster_index': [643, 213, 416, 637]})

analysis_mapping_K_space.append({'name': 'poor dough into pan',
                         'cluster_index': [439, 523, ]})
analysis_mapping_K_space.append({'name': 'open box ingredients', # TMP very small...
                         'cluster_index': [598 ]})
#analysis_mapping_K_space.append({'name': 'movement_recipe',
#                         'cluster_index': [401]})
analysis_mapping_K_space.append({'name': 'clean hands',
                         'cluster_index': [133, 267, 406, 579,]})
analysis_mapping_K_space.append({'name': 'use balance',
                         'cluster_index': [101, 259, 594, ]})
analysis_mapping_K_space.append({'name': 'look kitchen',
                         'cluster_index': [94]})
analysis_mapping_K_space.append({'name': 'look shelf',
                         'cluster_index': [69, 614]})
analysis_mapping_K_space.append({'name': 'use mixer',
                         'cluster_index': [49 ]})
analysis_mapping_K_space.append({'name': 'break chocolate',
                         'cluster_index': [525 ]}) # 51, 52 is adding chocolate from a box..
analysis_mapping_K_space.append({'name': 'merge white eggs',
                         'cluster_index': [529]})
analysis_mapping_K_space.append({'name': 'merge butter chocolate',
                         'cluster_index': [89, 350, 597, ]})
analysis_mapping_K_space.append({'name': 'poor flour/sugar',
                         'cluster_index': [9, 35, 314, 358, 383, 421, ]})
analysis_mapping_K_space.append({'name': 'use sink',
                         'cluster_index': [323]})
analysis_mapping_K_space.append({'name': 'manipulate oven',
                         'cluster_index': [208, 289, 349, ]})
adf_K_space = pd.DataFrame(analysis_mapping_K_space)
adf_K_space['pyr_cluster_index'] = adf_K_space['cluster_index'].apply(lambda x: ['R'+str(i) for i in x])


analysis_mapping_G_space = []

analysis_mapping_G_space.append({'name': 'recipe',
                         'cluster_index': [112,113]})
#analysis_mapping_G_space.append({'name': 'movement_recipe',
#                         'cluster_index': [112,113,  ]})
analysis_mapping_G_space.append({'name': 'prepare mixer',
                         'cluster_index': []})
analysis_mapping_G_space.append({'name': 'use mixer',
                         'cluster_index': [81, 82, 83, 8, ]})
analysis_mapping_G_space.append({'name': 'use balance',
                         'cluster_index': [69]})
analysis_mapping_G_space.append({'name': 'poor flour/sugar',
                         'cluster_index': [24, 85, 86, ]})
analysis_mapping_G_space.append({'name': 'talking',
                         'cluster_index': []})
analysis_mapping_G_space.append({'name': 'butter',
                         'cluster_index': [175 ]})
analysis_mapping_G_space.append({'name': 'merge flour/sugar',
                         'cluster_index': [19, 20 ]})
analysis_mapping_G_space.append({'name': 'look window',
                         'cluster_index': []})
analysis_mapping_G_space.append({'name': 'manipulate oven',
                         'cluster_index': [135, 139, 47]})
analysis_mapping_G_space.append({'name': 'setting oven',
                         'cluster_index': [138, 139]})
analysis_mapping_G_space.append({'name': 'merge white eggs',
                         'cluster_index': [34, 35, 56, ]})
analysis_mapping_G_space.append({'name': 'poor dough into pan',
                         'cluster_index': [43, 44, 45, 46,48 ]})
#analysis_mapping_G_space.append({'name': 'put_pan_to_manipulate oven',
#                         'cluster_index': [47 ]})
analysis_mapping_G_space.append({'name': 'open box ingredients', # TMP very small...
                         'cluster_index': [94 ]})
analysis_mapping_G_space.append({'name': 'exogeneous',
                         'cluster_index': []})
analysis_mapping_G_space.append({'name': 'look kitchen',
                         'cluster_index': [115]})
analysis_mapping_G_space.append({'name': 'look shelf',
                         'cluster_index': [126]})
adf_G_space = pd.DataFrame(analysis_mapping_G_space)
adf_G_space['pyr_cluster_index'] = adf_G_space['cluster_index'].apply(lambda x: ['K'+str(i) for i in x])


analysis_mapping_G_opt_space = []

analysis_mapping_G_opt_space.append({'name': 'recipe',
                         'cluster_index': [0, 1, 2, 3, 9, 10, 11, 12]})
#analysis_mapping_G_opt_space.append({'name': 'movement_recipe',
#                         'cluster_index': []})
analysis_mapping_G_opt_space.append({'name': 'plug mixer',
                         'cluster_index': [22]})
analysis_mapping_G_opt_space.append({'name': 'use measuring cup',
                         'cluster_index': [31]})
analysis_mapping_G_opt_space.append({'name': 'break chocolate',
                         'cluster_index': [32, 51, 52, 53, ]}) # 51, 52 is adding chocolate from a box...
analysis_mapping_G_opt_space.append({'name': 'open butter/chocolate',
                         'cluster_index': [33]})
analysis_mapping_G_opt_space.append({'name': 'merge butter chocolate',
                         'cluster_index': [47, 48, 69,73, 74, 75,  ]})
analysis_mapping_G_opt_space.append({'name': 'poor flour/sugar',
                         'cluster_index': [59, 61, ]})
analysis_mapping_G_opt_space.append({'name': 'merge flour/sugar',
                         'cluster_index': []})
analysis_mapping_G_opt_space.append({'name': 'merge white eggs',
                         'cluster_index': [70, 71, 72, ]})
analysis_mapping_G_opt_space.append({'name': 'butter',
                         'cluster_index': [34, 36, 37, 38,  46, ]})
analysis_mapping_G_opt_space.append({'name': 'break eggs',
                         'cluster_index': [39, 40, 41, 42, ]})
analysis_mapping_G_opt_space.append({'name': 'merge flour/eggs/sugar',
                         'cluster_index': [43, 44 ]})
analysis_mapping_G_opt_space.append({'name': 'use sink',
                         'cluster_index': [24, 25, 27, ]})
analysis_mapping_G_opt_space.append({'name': 'fridge',
                         'cluster_index': [4]})
analysis_mapping_G_opt_space.append({'name': 'prepare mixer',
                         'cluster_index': [54, 55]})
analysis_mapping_G_opt_space.append({'name': 'use mixer',
                         'cluster_index': [56, 57, 58, 65]}) #57 is a mix of setting up and use
analysis_mapping_G_opt_space.append({'name': 'manipulate oven',
                         'cluster_index': [6]})
analysis_mapping_G_opt_space.append({'name': 'look kitchen',
                         'cluster_index': [8]})
analysis_mapping_G_opt_space.append({'name': 'talking',
                         'cluster_index': [14]})
analysis_mapping_G_opt_space.append({'name': 'look window',
                         'cluster_index': [13]})
analysis_mapping_G_opt_space.append({'name': 'exogeneous',
                         'cluster_index': []})
adf_G_opt_space = pd.DataFrame(analysis_mapping_G_opt_space)

adf_G_opt_space['pyr_cluster_index'] = adf_G_opt_space['cluster_index'].apply(lambda x: ['G'+str(i) for i in x])

# Where is G58 ? 
mapping_category_df = {n: [] for n in set(adf_K_space['name']).union(set(adf_G_space['name'])).union(set(adf_G_opt_space['name']))}

for cat_name in mapping_category_df.keys():
    if len(adf_K_space.loc[adf_K_space.name == cat_name, 'pyr_cluster_index']) > 0:
        mapping_category_df[cat_name].extend(adf_K_space.loc[adf_K_space.name == cat_name, 'pyr_cluster_index'].iloc[0])
        
    if len(adf_G_space.loc[adf_G_space.name == cat_name, 'pyr_cluster_index']) > 0:
        mapping_category_df[cat_name].extend(adf_G_space.loc[adf_G_space.name == cat_name, 'pyr_cluster_index'].iloc[0])
        
    if len(adf_G_opt_space.loc[adf_G_opt_space.name == cat_name, 'pyr_cluster_index']) > 0:
        mapping_category_df[cat_name].extend(adf_G_opt_space.loc[adf_G_opt_space.name == cat_name, 'pyr_cluster_index'].iloc[0])

mapping_category_df = pd.DataFrame({
    'name': list(mapping_category_df.keys()),
    'cluster_index': list(mapping_category_df.values())
})


mapping_cluster_category = defaultdict(list)
for _, row in mapping_category_df.iterrows():
    cat = row['name']
    for cl in row['cluster_index']:
        mapping_cluster_category[cl].append(cat)
        
        
        
        
# from collections import defaultdict
# import pandas as pd


# analysis_mapping_K_space = []

# analysis_mapping_K_space.append({'name': 'recipe',
#                          'cluster_index': []})
# analysis_mapping_K_space.append({'name': 'movement_recipe',
#                          'cluster_index': []})
# analysis_mapping_K_space.append({'name': 'setting_batter',
#                          'cluster_index': []})
# analysis_mapping_K_space.append({'name': 'talking',
#                          'cluster_index': [137, 487, 352, 644]})
# analysis_mapping_K_space.append({'name': 'window',
#                          'cluster_index': []})
# analysis_mapping_K_space.append({'name': 'oven',
#                          'cluster_index': [379, 534, 266, 129]})
# analysis_mapping_K_space.append({'name': 'setting_oven',
#                          'cluster_index': [504, 97, 313]})
# analysis_mapping_K_space.append({'name': 'exogeneous',
#                          'cluster_index': [643, 213, 416, 637]})

# analysis_mapping_K_space.append({'name': 'poor_dough_pan',
#                          'cluster_index': [439, 523, ]})
# analysis_mapping_K_space.append({'name': 'open_box_ingredients', # TMP very small...
#                          'cluster_index': [598 ]})
# analysis_mapping_K_space.append({'name': 'movement_recipe',
#                          'cluster_index': [401]})
# analysis_mapping_K_space.append({'name': 'clean_hands',
#                          'cluster_index': [133, 267, 406, 579,]})
# analysis_mapping_K_space.append({'name': 'use_balance',
#                          'cluster_index': [101, 259, 594, ]})
# analysis_mapping_K_space.append({'name': 'look_kitchen',
#                          'cluster_index': [94]})
# analysis_mapping_K_space.append({'name': 'look_etagere',
#                          'cluster_index': [69, 614]})
# analysis_mapping_K_space.append({'name': 'use_batter',
#                          'cluster_index': [49 ]})
# analysis_mapping_K_space.append({'name': 'breaking_chocolate',
#                          'cluster_index': [525 ]}) # 51, 52 is adding chocolate from a box..
# analysis_mapping_K_space.append({'name': 'merge_white_chocolate',
#                          'cluster_index': [529]})
# analysis_mapping_K_space.append({'name': 'merge_hand_chocolate_dough',
#                          'cluster_index': [89, 350, 597, ]})
# analysis_mapping_K_space.append({'name': 'poor_flour_sugar',
#                          'cluster_index': [9, 35, 314, 358, 383, 421, ]})
# analysis_mapping_K_space.append({'name': 'sink',
#                          'cluster_index': [323]})
# analysis_mapping_K_space.append({'name': 'oven',
#                          'cluster_index': [208, 289, 349, ]})
# adf_K_space = pd.DataFrame(analysis_mapping_K_space)
# adf_K_space['pyr_cluster_index'] = adf_K_space['cluster_index'].apply(lambda x: ['R'+str(i) for i in x])


# analysis_mapping_G_space = []

# analysis_mapping_G_space.append({'name': 'recipe',
#                          'cluster_index': []})
# analysis_mapping_G_space.append({'name': 'movement_recipe',
#                          'cluster_index': [112,113,  ]})
# analysis_mapping_G_space.append({'name': 'setting_batter',
#                          'cluster_index': []})
# analysis_mapping_G_space.append({'name': 'use_batter',
#                          'cluster_index': [81, 82, 83, 8, ]})
# analysis_mapping_G_space.append({'name': 'use_balance',
#                          'cluster_index': [69]})
# analysis_mapping_G_space.append({'name': 'poor_flour_sugar',
#                          'cluster_index': [24, 85, 86, ]})
# analysis_mapping_G_space.append({'name': 'talking',
#                          'cluster_index': []})
# analysis_mapping_G_space.append({'name': 'butter',
#                          'cluster_index': [175 ]})
# analysis_mapping_G_space.append({'name': 'merge_flour_sugar',
#                          'cluster_index': [19, 20 ]})
# analysis_mapping_G_space.append({'name': 'window',
#                          'cluster_index': []})
# analysis_mapping_G_space.append({'name': 'oven',
#                          'cluster_index': [135, 139, ]})
# analysis_mapping_G_space.append({'name': 'setting_oven',
#                          'cluster_index': [138, 139]})
# analysis_mapping_G_space.append({'name': 'merge_white_chocolate',
#                          'cluster_index': [34, 35, 56, ]})
# analysis_mapping_G_space.append({'name': 'poor_dough_pan',
#                          'cluster_index': [43, 44, 45, 46,48 ]})
# analysis_mapping_G_space.append({'name': 'put_pan_to_oven',
#                          'cluster_index': [47 ]})
# analysis_mapping_G_space.append({'name': 'open_box_ingredients', # TMP very small...
#                          'cluster_index': [94 ]})
# analysis_mapping_G_space.append({'name': 'exogeneous',
#                          'cluster_index': []})
# analysis_mapping_G_space.append({'name': 'look_kitchen',
#                          'cluster_index': [115]})
# analysis_mapping_G_space.append({'name': 'look_etagere',
#                          'cluster_index': [126]})
# adf_G_space = pd.DataFrame(analysis_mapping_G_space)
# adf_G_space['pyr_cluster_index'] = adf_G_space['cluster_index'].apply(lambda x: ['K'+str(i) for i in x])


# analysis_mapping_G_opt_space = []

# analysis_mapping_G_opt_space.append({'name': 'recipe',
#                          'cluster_index': [0, 1, 2, 3, 9, 10, 11, 12]})
# analysis_mapping_G_opt_space.append({'name': 'movement_recipe',
#                          'cluster_index': []})
# analysis_mapping_G_opt_space.append({'name': 'plugging',
#                          'cluster_index': [22]})
# analysis_mapping_G_opt_space.append({'name': 'graduator',
#                          'cluster_index': [31]})
# analysis_mapping_G_opt_space.append({'name': 'breaking_chocolate',
#                          'cluster_index': [32, 51, 52, 53, ]}) # 51, 52 is adding chocolate from a box...
# analysis_mapping_G_opt_space.append({'name': 'open_butter_chocolate',
#                          'cluster_index': [33]})
# analysis_mapping_G_opt_space.append({'name': 'merge_hand_chocolate_dough',
#                          'cluster_index': [47, 48, 69,73, 74, 75,  ]})
# analysis_mapping_G_opt_space.append({'name': 'poor_flour_sugar',
#                          'cluster_index': [59, 61, ]})
# analysis_mapping_G_opt_space.append({'name': 'merge_flour_sugar',
#                          'cluster_index': []})
# analysis_mapping_G_opt_space.append({'name': 'merge_white_chocolate',
#                          'cluster_index': [70, 71, 72, ]})
# analysis_mapping_G_opt_space.append({'name': 'butter',
#                          'cluster_index': [34, 36, 37, 38,  46, ]})
# analysis_mapping_G_opt_space.append({'name': 'break eggs',
#                          'cluster_index': [39, 40, 41, 42, ]})
# analysis_mapping_G_opt_space.append({'name': 'hand_merge_eggs_flour',
#                          'cluster_index': [43, 44 ]})
# analysis_mapping_G_opt_space.append({'name': 'sink',
#                          'cluster_index': [24, 25, 27, ]})
# analysis_mapping_G_opt_space.append({'name': 'fridge',
#                          'cluster_index': [4]})
# analysis_mapping_G_opt_space.append({'name': 'setting_batter',
#                          'cluster_index': [54, 55]})
# analysis_mapping_G_opt_space.append({'name': 'use_batter',
#                          'cluster_index': [56, 57, 58, 65]}) #57 is a mix of setting up and use
# analysis_mapping_G_opt_space.append({'name': 'oven',
#                          'cluster_index': [6]})
# analysis_mapping_G_opt_space.append({'name': 'look_kitchen',
#                          'cluster_index': [8]})
# analysis_mapping_G_opt_space.append({'name': 'talking',
#                          'cluster_index': [14]})
# analysis_mapping_G_opt_space.append({'name': 'window',
#                          'cluster_index': [13]})
# analysis_mapping_G_opt_space.append({'name': 'exogeneous',
#                          'cluster_index': []})
# adf_G_opt_space = pd.DataFrame(analysis_mapping_G_opt_space)

# adf_G_opt_space['pyr_cluster_index'] = adf_G_opt_space['cluster_index'].apply(lambda x: ['G'+str(i) for i in x])

# # Where is G58 ? 
# mapping_category_df = {n: [] for n in set(adf_K_space['name']).union(set(adf_G_space['name'])).union(set(adf_G_opt_space['name']))}

# for cat_name in mapping_category_df.keys():
#     if len(adf_K_space.loc[adf_K_space.name == cat_name, 'pyr_cluster_index']) > 0:
#         mapping_category_df[cat_name].extend(adf_K_space.loc[adf_K_space.name == cat_name, 'pyr_cluster_index'].iloc[0])
        
#     if len(adf_G_space.loc[adf_G_space.name == cat_name, 'pyr_cluster_index']) > 0:
#         mapping_category_df[cat_name].extend(adf_G_space.loc[adf_G_space.name == cat_name, 'pyr_cluster_index'].iloc[0])
        
#     if len(adf_G_opt_space.loc[adf_G_opt_space.name == cat_name, 'pyr_cluster_index']) > 0:
#         mapping_category_df[cat_name].extend(adf_G_opt_space.loc[adf_G_opt_space.name == cat_name, 'pyr_cluster_index'].iloc[0])

# mapping_category_df = pd.DataFrame({
#     'name': list(mapping_category_df.keys()),
#     'cluster_index': list(mapping_category_df.values())
# })


# mapping_cluster_category = defaultdict(list)
# for _, row in mapping_category_df.iterrows():
#     cat = row['name']
#     for cl in row['cluster_index']:
#         mapping_cluster_category[cl].append(cat)
        