#false_pos_neg_df.py
#generate panda dataframe for false positive and false negative analysis

import pandas as pd
import json 

all_dfs_fn = []
all_dfs_fp = []
for i in range(5):
    with open(f'./models/kfold_cnn_rgbs_5l/fold_{i+1}/misclassification/false_positives.json') as f:
        fp_data = json.load(f)
    with open(f'./models/kfold_cnn_rgbs_5l/fold_{i+1}/misclassification/false_negatives.json') as f:
        fn_data = json.load(f)
    fn_data = pd.DataFrame(fn_data)
    fp_data = pd.DataFrame(fp_data)

    fn_data['fold'] = i+1
    fp_data['fold'] = i+1
    all_dfs_fn.append(fn_data)
    all_dfs_fp.append(fp_data)

#concatenate all dataframes
final_fp_df = pd.concat(all_dfs_fp, ignore_index=True)
final_fn_df = pd.concat(all_dfs_fn, ignore_index=True)

#sort by predicted probability
final_fp_df.sort_values(by='predicted_probability', ascending=False, inplace=True)
final_fn_df.sort_values(by='predicted_probability', ascending=True, inplace=True)

#reset index
final_fn_df.reset_index(drop=True, inplace=True)
final_fp_df.reset_index(drop=True, inplace=True)

#save to csv
final_fn_df.to_csv('./models/kfold_cnn_rgbs_5l/all_false_negatives.csv', index=False)
final_fp_df.to_csv('./models/kfold_cnn_rgbs_5l/all_false_positives.csv', index=False)

print("False negatives and false positives dataframes saved to CSV.")

