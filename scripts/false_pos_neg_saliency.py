#false_pos_neg_saliency.py
#create saliency maps for false positive and false negative images
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.gradcam_utils import contrastive_saliency, show_contrastive_saliency, load_tensor_rgbs
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


#load false positive and false negative dataframes
fn_df = pd.read_csv('./models/kfold_cnn_rgbs_5l/all_false_negatives.csv')
fp_df = pd.read_csv('./models/kfold_cnn_rgbs_5l/all_false_positives.csv')

def generate_saliency_maps(df, output_subdir):
    
    for index, row in df.iterrows():
        img_path = row['image_path']
        model_dir = f"./models/kfold_cnn_rgbs_5l/fold_{row['fold']}"
        model_path = os.path.join(model_dir, "model.keras")
        
        # Load model
       
        model = load_model(model_path)
        
        # Load image
        picture_size = 128
        img, x = load_tensor_rgbs(img_path, picture_size)
        
        # Generate saliency map
        contrast, grad_pos, grad_neg = contrastive_saliency(model, x, target_size=picture_size)
    
        prob = row['predicted_probability']

        # Save saliency map
        save_dir = os.path.join(model_dir, "explainability", "saliency_maps")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{os.path.basename(img_path).split('.')[0]}_saliency.png")
        show_contrastive_saliency(x, contrast, prob, save_path=save_path, show=False)
        print(f"Saved saliency map to {save_path}")

        #add saliency image path to dataframe
        df.at[index, 'saliency_map_path'] = save_path

        #clear resources
        del contrast, x, img
        #clear keras session
        K.clear_session()
       
    return df
        

# Generate saliency maps for false negatives
new_fn_df = generate_saliency_maps(fn_df, 'false_negatives')

# Generate saliency maps for false positives
new_fp_df = generate_saliency_maps(fp_df, 'false_positives')

#save updated dataframes with saliency map paths
new_fn_df.to_csv('./models/kfold_cnn_rgbs_5l/false_negatives_with_saliency.csv', index=False)
new_fp_df.to_csv('./models/kfold_cnn_rgbs_5l/false_positives_with_saliency.csv', index=False)

