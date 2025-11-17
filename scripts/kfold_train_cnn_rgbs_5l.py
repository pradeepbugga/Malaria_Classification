import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_with_tf_dataset_tensor_for_kfold
from src.model_builder import cnn_rgbs_5l
from src.train_utils import train_and_save_kfold
import numpy as np

import tensorflow as tf
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

#set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1' # Try to force deterministic ops on GPU
random.seed(42)  
np.random.seed(42)
tf.random.set_seed(42)  

data_dir = './data/cell_images'
output_dir='./models/kfold_cnn_rgbs_5l'
class_order = ['uninfected','parasitized']

#generate ND array of all paths and labels combined for k-fold splitting
all_paths, all_labels = [], []
for i, cls in enumerate(class_order):
    for split in ['train', 'test']:
        folder = os.path.join(data_dir, split, cls)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png'))]
        all_paths.extend(files)
        all_labels.extend([i] * len(files))

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)  

all_paths, all_labels = shuffle(all_paths, all_labels, random_state=42)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []   


for fold, (train_index, val_index) in enumerate(kf.split(all_paths, all_labels)):
    print(f"Starting fold {fold + 1}")
    train_paths, val_paths = all_paths[train_index], all_paths[val_index]
    train_labels, val_labels = all_labels[train_index], all_labels[val_index]

    print("Train class balance:", np.mean(train_labels))
    print("Val class balance:", np.mean(val_labels))

    train_ds = load_with_tf_dataset_tensor_for_kfold(train_paths, train_labels, augment=True)
    val_ds = load_with_tf_dataset_tensor_for_kfold(val_paths, val_labels, augment=False)
    
    model = cnn_rgbs_5l(picture_size=128)
    train_and_save_kfold(model, train_ds, val_ds, all_paths, all_labels, val_index, os.path.join(output_dir, f'fold_{fold + 1}'), fold + 1, learning_rate=0.001, use_EarlyStopping=True, use_ReduceLROnPlateau=True, 
            es_patience=6, rlrop_patience=3, epochs=30)
    print(f"Completed fold {fold + 1}\n")
