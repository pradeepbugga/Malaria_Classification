#visualize_clahe.py
#this script visualizes the effect of CLAHE preprocessing on sample images
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.visualize import plot_hsv
img_path = "./data/cell_images/test/uninfected/C68P29N_ThinF_IMG_20150819_134504_cell_167.png"

if "parasitized" in img_path:
    model_dir = "./models/test3/samples/parasitized"
elif "uninfected" in img_path:
    model_dir = "./models/test3/samples/uninfected"
else:
    print("Error: Cannot determine class from image path.")
   
plot_hsv(img_path, save_path=os.path.join(model_dir, "hsv_visualization.png"), composite=True, split_channels=True, RGB_visualization=True)