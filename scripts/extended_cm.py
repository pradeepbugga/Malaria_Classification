#extended_cm.py
#this scripts plots extended confusion matrix from prediction probabilities
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analyze import extended_confusion_matrix

model_dir = './models/kfold_cnn_rgbs_5l/fold_5'
extended_confusion_matrix(model_dir)
