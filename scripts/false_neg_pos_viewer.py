#false_neg_pos_viewer.py
#this script visualizes false negative and false positive images based on model predictions
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analyze import view_images_by_probability

model_dir = '/home/pb929/Projects/Malaria_Classification/models/cnn_rgbh_5l_es_pat6_rlrop_pat3'
view_images_by_probability(model_dir, bin_range=(0, 0.1), true_label=1, max_images=5)  # False Negatives    
#view_images_by_probability(model_dir, bin_range=(0.9, 1.0), true_label=0, max_images=5)  # False Positives    

