#analyze.py
#this scripts analyzes model prediction probabilities and visualizes images in specific probability ranges

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os
import json

def extended_confusion_matrix(model_dir):
    
    predictions_csv_path = os.path.join(model_dir, "predictions", "predictions.csv")
    df=pd.read_csv(predictions_csv_path)

    # make sure df has the 'bin' column
    df["bin"] = pd.cut(df["y_prob"], bins=np.arange(0, 1.1, 0.1))

    counts = df.groupby(["y_true", "bin"]).size().unstack(fill_value=0)
    sns.heatmap(counts, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Probability Bin')
    plt.title('Extended Confusion Matrix')
    plt.savefig(os.path.join(model_dir, "reports", "extended_confusion_matrix.png"), bbox_inches="tight")
    plt.show()

def view_images_by_probability(model_dir, bin_range=(0.2, 0.3), true_label=1, max_images=5):
    
    predictions_csv_path = os.path.join(model_dir, "predictions", "predictions.csv")
    df=pd.read_csv(predictions_csv_path)

    # make sure df has the 'bin' column
    df["bin"] = pd.cut(df["y_prob"], bins=np.arange(0, 1.1, 0.1))

    ## Select the interval object for (0.0, 0.1]
    target_bin = pd.Interval(bin_range[0], bin_range[1], closed='right')

    subset = df[(df["y_true"] == true_label) & (df["bin"] == target_bin)]

    print("This bin has ", subset.shape[0], " images.")

    # view a few examples
    
    columns = 5
    rows = max_images // columns + int(max_images % columns > 0)
    
    x=1
    for i, row in subset.head(max_images).iterrows():
        
        img_path = f"{row['filename']}"  
        print(img_path)
        img = mpimg.imread(img_path)
        if img is None:
            print(f"Image at {img_path} could not be loaded.")
            continue
        plt.subplot(rows,columns,x)
        plt.imshow(img)
        
        plt.title(f"Prob={row['y_prob']:.2f}")
        plt.axis('off')
        x+=1

    plt.tight_layout()    
    plt.show()

def log_false_negatives_positives(model_dir):
    
    # log false negatives and false positives to separate log files

    predictions_csv_path = os.path.join(model_dir, "predictions", "predictions.csv")
    df=pd.read_csv(predictions_csv_path)

    #false negative subset
    subset_fn = df[(df["y_true"] == 1) & (df["y_pred"] <= 0.5)]
    

    #false positive subset
    subset_fp = df[(df["y_true"] == 0) & (df["y_pred"] > 0.5)]

    false_negatives = []
    false_positives = []

    for i, row in subset_fn.iterrows():
        
        #write full path, true label, predicted label, and probability to json file
        image_path = f"{row['filename']}"
        index = image_path.find('/data')
        image_path = image_path[index:]  # trim to start from /data
        image_path = '.' + image_path  # make it relative path
        

        false_negatives.append({
            "image_path": image_path,
            "true_label": int(row['y_true']),
            "predicted_label": int(row['y_pred']),
            "predicted_probability": float(row['y_prob'])
        })

           
    for i, row in subset_fp.iterrows():
        image_path = f"{row['filename']}"
        index = image_path.find('/data')
        image_path = image_path[index:]  # trim to start from /data
        image_path = '.' + image_path  # make it relative path

        #write full path, true label, predicted label, and probability to json file
        false_positives.append({
            "image_path": f"{row['filename']}",
            "true_label": int(row['y_true']),
            "predicted_label": int(row['y_pred']),
            "predicted_probability": float(row['y_prob'])
        })
    
    #sort by predicted probability
    false_negatives = sorted(false_negatives, key=lambda x: x['predicted_probability'])
    false_positives = sorted(false_positives, key=lambda x: x['predicted_probability'], reverse=True)   

    #save to json files
    os.makedirs(os.path.join(model_dir, "misclassification"), exist_ok=True)
    with open(os.path.join(model_dir, "misclassification", 'false_negatives.json'), 'w') as fn_file:
        json.dump(false_negatives, fn_file, indent=4)   
    with open(os.path.join(model_dir, "misclassification", 'false_positives.json'), 'w') as fp_file:
        json.dump(false_positives, fp_file, indent=4)


