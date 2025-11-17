#run_saliency.py
# This script runs the saliency map visualization on a given image to show model reasoning

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2

from src.gradcam_utils import saliency_rgb_only_sidebyside  # import your function

if __name__ == "__main__":
    model_dir = "./models/kfold_cnn_rgbs_5l/fold_1"
    model_path = os.path.join(model_dir, "model.keras")
    img_path = "./data/cell_images/train/parasitized/C180P141NThinF_IMG_20151201_163848_cell_140.png"
    
    # ---- Load model ----
    model = load_model(model_path)
    picture_size = 128

    # ---- Load image with 4 channels (RGB+S) ----
    def load_tensor_rgbs(path, target_size=128):
        img = load_img(path, target_size=(target_size, target_size))
        rgb = img_to_array(img).astype("float32") / 255.0
        hsv = tf.image.rgb_to_hsv(rgb)
        S = hsv[:, :, 1:2]
        rgbs = tf.concat([rgb, S], axis=-1)
        return img, np.expand_dims(rgbs.numpy(), 0)

    img, x = load_tensor_rgbs(img_path, picture_size)

    # ---- Run saliency ----
    sal_map = saliency_rgb_only_sidebyside(model, x, target_class=0, save_path=os.path.join(model_dir, "explainability", "saliency_side_by_side.png"))

