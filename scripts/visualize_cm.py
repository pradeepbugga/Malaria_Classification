# visualize_cm.py
#this file contains functions to visualize confusion matrix

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.visualize import plot_confusion_matrix

model_dir = "/home/pb929/Projects/Malaria_Classification/models/cnn_rgbs_5l"
plot_confusion_matrix(model_dir, save_path=os.path.join(model_dir, "reports", "confusion_matrix.png"))