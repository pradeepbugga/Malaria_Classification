import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_with_imagedatagenerator
from src.model_builder import cnn_rgb_3l
from src.train_utils import train_and_save

import tensorflow as tf
import random

#set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1' # Try to force deterministic ops on GPU
random.seed(42)  
np.random.seed(42)
tf.random.set_seed(42) 

data_dir = './data/cell_images'
output_dir='./models/test'

train_gen, val_gen, (test_gen, test_filenames) = load_with_imagedatagenerator(data_dir, augment=True)
model = cnn_rgb_3l(picture_size=128)
train_and_save(model, train_gen, val_gen, test_gen, test_filenames, output_dir, learning_rate=0.001, use_EarlyStopping=True, use_ReduceLROnPlateau=False, es_patience=2, rlrop_patience=0, epochs=10)
