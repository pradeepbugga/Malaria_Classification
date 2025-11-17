#run_gradcam.py
# This script runs the Grad-CAM visualization on a given image using a pre-trained model

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gradcam_utils import generate_gradcam

if __name__ == "__main__":
    model_dir = "./models/cnn_rgbs_5l"
    model_path = os.path.join(model_dir, "model.keras")
    img_path = "./data/cell_images/test/uninfected/C68P29N_ThinF_IMG_20150819_134504_cell_167.png"
    output_path = os.path.join(model_dir, "gradcam_output_fp2.png")

    prob = generate_gradcam(model_path, img_path, last_conv_name="conv2d_4", output_path=output_path)
    print(f"Grad-CAM saved → {output_path}")
    print(f"Predicted probability: {prob:.3f}")