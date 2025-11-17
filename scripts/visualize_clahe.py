#visualize_clahe.py
#this script visualizes the effect of CLAHE preprocessing on sample images
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.visualize import plot_clahe
img_path = "./data/cell_images/test/parasitized/C39P4thinF_original_IMG_20150622_111206_cell_75.png"

if "parasitized" in img_path:
    model_dir = "./models/cnn_rgb_clahe_3l/samples/parasitized"
elif "uninfected" in img_path:
    model_dir = "./models/cnn_rgb_clahe_3l/samples/uninfected"
else:
    print("Error: Cannot determine class from image path.")
   
plot_clahe(img_path, save_path=os.path.join(model_dir, "clahe_visualization.png"))