#run_contrastive_saliency.py
# This script runs the contrastive salience visualization on a given image using a pre-trained model

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

from src.gradcam_utils import contrastive_saliency, show_contrastive_saliency, load_tensor_rgbs

if __name__ == "__main__":
    model_dir = "./models/cnn_rgbs_5l/"
    model_path = os.path.join(model_dir, "model.keras")
    img_path = "./data/cell_images/test/parasitized/C39P4thinF_original_IMG_20150622_111206_cell_46.png"
    
    # ---- Load model ----
    model = load_model(model_path)
    picture_size = 128
    img, x = load_tensor_rgbs(img_path, picture_size)

    # ---- Run saliency ----
    contrast, grad_pos, grad_neg, prob = contrastive_saliency(model, x, target_size = picture_size)
    show_contrastive_saliency(x, contrast, prob, save_path=os.path.join(model_dir, "explainability", "fn1_contrastive_saliency.png"))
