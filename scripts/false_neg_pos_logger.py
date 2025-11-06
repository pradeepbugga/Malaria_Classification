#false_neg_pos_logger.py
#this script visualizes false negative and false positive images based on model predictions
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analyze import log_false_negatives_positives

model_dir = '/home/pb929/Projects/Malaria_Classification/models/cnn_rgbs_5l'
log_false_negatives_positives(model_dir)
