#visualize.py
#this file contains functions to visualize history, confusion matrix, and image preprocessing


def plot_history(history_json_path, save_path=None):
    #this function plots training and validation accuracy and loss from a history json file
    import json
    import matplotlib.pyplot as plt
    with open(history_json_path, "r") as f:
        history = json.load(f)


    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")
    if save_path:
        plt.savefig(save_path.replace('.png', '_accuracy.png'))
    plt.show()
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")
    if save_path:
        plt.savefig(save_path.replace('.png', '_loss.png'))
    plt.show()

def plot_confusion_matrix(model_dir, save_path=None):
    #this function plots a confusion matrix
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os

    class_names = ['uninfected', 'parasitized']
    cm = np.load(os.path.join(model_dir, "reports", "confusion_matrix.npy"))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_clahe(img_path, save_path=None):
    #this function visualizes the effect of CLAHE augmentation on an image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    
    def clahe_aug_augmentation(img_rgb_float):
        """Applies CLAHE to the V-channel (luminance) and converts back to RGB."""
        
        # 1. Convert Keras float [0, 1] to OpenCV uint8 [0, 255]
        img_uint8 = (img_rgb_float * 255).astype(np.uint8) 
        
        # 2. Convert to LAB or HSV (LAB is often preferred for preservation)
        img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB) 
        L, A, B = cv2.split(img_lab)

        # 3. Apply CLAHE ONLY to the L (Luminance) channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)) # Use your optimized params
        L_enhanced = clahe.apply(L)

        # 4. Merge back and convert to RGB
        img_merged_lab = cv2.merge([L_enhanced, A, B])
        img_output_rgb = cv2.cvtColor(img_merged_lab, cv2.COLOR_LAB2RGB)

        # 5. Convert back to Keras float [0, 1]
        return img_output_rgb.astype(np.float32) / 255.0
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #CV2 reads in BGR format
    clahe_img = clahe_aug_augmentation(img)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original RGB Image")
    plt.axis("off")    
    plt.subplot(1, 2, 2)
    plt.imshow(clahe_img)
    plt.title("CLAHE Augmented Image")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.show()  

def plot_hsv(img_path, save_path, composite = True, split_channels = False, RGB_visualization = True):
    # this function visualizes the HSV conversion of an image
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2, os
    import numpy as np

    #create folder if doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

    rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #CV2 reads in BGR format
    hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    # Decide number of subplots
    n_columns = 1 + (1 if composite else 0) + (3 if split_channels else 0)
    fig, axes = plt.subplots(1, n_columns, figsize=(4 * n_columns, 4))

    if n_columns == 1:
        axes = [axes]  # make iterable for single-panel case

    # 1. RGB image
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    idx = 1

    # 2. HSV composite
    if composite:
        axes[idx].imshow(hsv_img)
        axes[idx].set_title("HSV composite")
        axes[idx].axis("off")
        idx += 1

    # 3. Split H, S, V
    if split_channels:
        for channel,title in zip([H, S, V], ["Hue", "Saturation", "Value"]):
            if RGB_visualization:
                axes[idx].imshow(channel)
            else:
                axes[idx].imshow(channel, cmap='gray')
            axes[idx].set_title(title)
            axes[idx].axis("off")
            idx += 1

    plt.tight_layout()
    plt.savefig(save_path)
    

def plot_rgbs(img_path, save_path=None):
    # this function visualizes the RGB+S image used in the RGBS model
    # Note: the model will see the image as 4 individual channels, but for visualization we can show it as a 3-channel image by duplicating the S channel

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    picture_size = 128
    def load_tensor_rgbs(path):
        """
        Loads an image from the given path and converts it to a tensor with RGBS channels.
        Returns the original image and the processed tensor.
        """
        # load and preprocess image
        img = load_img(path, target_size=(picture_size, picture_size))
        arr = img_to_array(img)                # [H,W,3] uint8
        arr = tf.cast(arr, tf.float32) / 255.0 # normalize 0-1

        # identical to training map_func
        img_hsv = tf.image.rgb_to_hsv(arr)
        s_channel = img_hsv[:, :, 1:2]         # keep dims
        rgbs = tf.concat([arr, s_channel], axis=-1)  # [H,W,4]

        # expand batch
        return rgbs

    y = load_tensor_rgbs(img_path)
    if y is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
    img = np.array(y)
    plt.subplot(1, 2, 1)
    rgb_img = load_img(img_path, target_size=(picture_size, picture_size))
    plt.imshow(rgb_img)
    plt.title("Original RGB Image")
    plt.axis("off")    
    plt.subplot(1, 2, 2)

    plt.imshow(img)
    plt.title("RGB+S Image")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path) 
    plt.show()

def write_channel_csv(img_path, channel, save_path):
    #this function generates a csv file containing pixel values of a specified channel from an image

    import cv2
    import numpy as np
    import csv
    import os


    img = cv2.imread(img_path)
    if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

    rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #CV2 reads in BGR format
    hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)

    # Map channel letter to array
    if channel == "H":
        arr = H
    elif channel == "S":
        arr = S
    elif channel == "V":
        arr = V
    else:
        raise ValueError("Channel must be one of 'H', 'S', or 'V'.")

     # Build new filename with channel tag before extension
    base, ext = os.path.splitext(save_path)
    save_path = f"{base}_{channel}{ext or '.csv'}"

    with open(save_path, 'w', newline='') as csvfile:
        csv_writer=csv.writer(csvfile)
        csv_writer.writerows(arr)








