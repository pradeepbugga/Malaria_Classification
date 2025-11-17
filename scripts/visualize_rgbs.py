#visualize_rgbs.py
#this script visualizes the effect of RGBS preprocessing on sample images
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.visualize import plot_rgbs
img_path = "./data/cell_images/test/parasitized/C39P4thinF_original_IMG_20150622_111206_cell_75.png"

if "parasitized" in img_path:
    model_dir = "./models/cnn_rgbs_5l/samples/parasitized"
elif "uninfected" in img_path:
    model_dir = "./models/cnn_rgbs_5l/samples/uninfected"
else:
    print("Error: Cannot determine class from image path.")
   
plot_rgbs(img_path, save_path=os.path.join(model_dir, "rgbs_visualization.png"))