#extended_cm.py
#this scripts plots extended confusion matrix from prediction probabilities
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analyze import extended_confusion_matrix

model_dir = '/home/pb929/Projects/Malaria_Classification/models/cnn_rgbs_5l'
extended_confusion_matrix(model_dir)
