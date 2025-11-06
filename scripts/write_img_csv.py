#write_img_csv.py
#this script extracts a specified channel from an image and writes the pixel values to a CSV file

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.visualize import write_channel_csv

img_path = "/home/pb929/Projects/Malaria_Classification/data/cell_images/test/parasitized/C39P4thinF_original_IMG_20150622_113842_cell_38.png"

model_dir = "/home/pb929/Projects/Malaria_Classification/models/cnn_rgbs_5l"
write_channel_csv(img_path, channel="S", save_path=os.path.join(model_dir, "channel_output.csv"))     


