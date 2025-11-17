#false_neg_pos_logger.py
#this script visualizes false negative and false positive images based on model predictions
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analyze import log_false_negatives_positives

model_dir = './models/kfold_cnn_rgbs_5l/fold_5'  # adjust as needed
log_false_negatives_positives(model_dir)
