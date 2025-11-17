# visualize_cm.py
#this file contains functions to visualize confusion matrix

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.visualize import plot_confusion_matrix

model_dir = "./models/kfold_cnn_rgbs_5l/fold_5"
plot_confusion_matrix(model_dir, save_path=os.path.join(model_dir, "reports", "confusion_matrix.png"))