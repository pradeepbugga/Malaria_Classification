#data_loader.py
#this file will load images from directories and create data generators for training, validation, and testing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math
import numpy as np
import cv2
from tensorflow.keras.utils import image_dataset_from_directory

def load_with_imagedatagenerator(data_dir, picture_size=128, batch_size=32, class_order=None, augment = False, preprocessing_fn=None):
    if class_order is None:
        class_order = ['uninfected','parasitized']
    
    if augment:
        #create augmented generator
        train_datagen=ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=90,  #rotate up to 90 degrees as cells are circular
            horizontal_flip=True,  #include horizontal flip
            vertical_flip=True,  #include vertical flip
            zoom_range=0.1,   #apply slight zooming (+/- 0.1)
            preprocessing_function=preprocessing_fn)
    else: 
        # no augmentation
        train_datagen=ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2, preprocessing_function=preprocessing_fn
            )   #apply slight zooming (+/- 0.1)

    test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocessing_fn)

    #use flow_from_directory to connect to folders
    train_generator = train_datagen.flow_from_directory(    
        directory=f'{data_dir}/train',
        target_size=(picture_size, picture_size),
        batch_size=batch_size,
        classes=class_order,
        class_mode='binary',   # 'binary' for two classes (0 and 1)
        subset='training',
        shuffle=True
    )
    val_generator=train_datagen.flow_from_directory(   #keras ignores augmentation if subset = validation
        directory=f'{data_dir}/train',
        target_size=(picture_size, picture_size),
        batch_size=batch_size,
        classes=class_order,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    test_generator=test_datagen.flow_from_directory(  #test set not augmented
        directory=f'{data_dir}/test',
        target_size=(picture_size, picture_size),
        batch_size=batch_size,
        classes=class_order,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, (test_generator, test_generator.filenames)

def load_with_tf_datagenerator(data_dir, picture_size=128, batch_size=32, class_order=None, augment= False, preprocessing_fn=None):
    if class_order is None:
        class_order = ['uninfected','parasitized']
    
    #train dataset
    train_ds_raw = image_dataset_from_directory(
        directory=f'{data_dir}/train',
        labels='inferred',
        label_mode='binary',
        image_size=(picture_size, picture_size),
        batch_size=None,
        subset='training',
        class_names=class_order,
        seed=42,
        validation_split=0.2
    )
      
    #validation dataset  
    val_ds_raw = image_dataset_from_directory(
        directory=f'{data_dir}/train',
        labels='inferred',
        label_mode='binary',
        image_size=(picture_size, picture_size),
        batch_size=None,
        subset='validation',
        class_names=class_order,
        seed=42,
        validation_split=0.2
    )

    #test dataset
    test_ds_raw = image_dataset_from_directory(
        directory=f'{data_dir}/test',
        labels='inferred',
        label_mode='binary',
        image_size=(picture_size, picture_size),
        batch_size=None,
        class_names=class_order,
        shuffle=False
        )
    
        # --- Augmentation function (Apply to training set only) ---
    def augment_func(image, label):
        # Rotation (The tf.image function handles the random angle)
        image = tf.image.rot90(
            image, 
            k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        )
        # Flips
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        return image, label

    full_paths = test_ds_raw.file_paths
    test_filenames = [
        os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p))
        for p in full_paths]


    def tf_rgbs_map_func(image, label):
        """
        Performs RGB to RGBS conversion using only TensorFlow operations.
        This replaces numpy_to_rgbh and map_func to eliminate RAM overhead.
        """
        
        # 1. Normalize RGB (Input is uint8, output is float [0, 1])
        image_float = tf.cast(image, tf.float32) / 255.0

        # 2. Convert to HSV using native TensorFlow
        # Output is float [0, 1] for H, S, V.
        img_hsv = tf.image.rgb_to_hsv(image_float) 
        
        # 3. Extract S channel (S is the second channel)
        # Use slicing [:, :, 1:2] to maintain the channel dimension (H, W, 1)
        S_normalized_tf = img_hsv[:, :, 1:2] 

        # 4. Stack RGB (3 channels) + S (1 channel)
        # tf.concat works efficiently on the GPU/CPU without creating NumPy copies.
        rgbs_image = tf.concat([image_float, S_normalized_tf], axis=-1)
        
        return rgbs_image, label

    def final_pipeline(ds, is_training=False):
        # 1. Convert RGB to RGBS
        ds = ds.map(tf_rgbs_map_func)

        # 2. Add augmentation for training set
        if is_training and augment:
            ds = ds.map(augment_func) 
        
        ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    #build final datsets
    train_ds = final_pipeline(train_ds_raw, is_training=True)
    val_ds = final_pipeline(val_ds_raw, is_training=False)
    test_ds = final_pipeline(test_ds_raw, is_training=False)


    return train_ds, val_ds, (test_ds, test_filenames)

def load_for_dual_rgb_hsv(data_dir, picture_size=128, batch_size=32, class_order=None, augment= False, preprocessing_fn=None):
    if class_order is None:
        class_order = ['uninfected','parasitized']

    if augment:
        #create augmented generator
        train_datagen=ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=90,  #rotate up to 90 degrees as cells are circular
            horizontal_flip=True,  #include horizontal flip
            vertical_flip=True,  #include vertical flip
            zoom_range=0.1,   #apply slight zooming (+/- 0.1)
            preprocessing_function=preprocessing_fn)
    else: 
        # no augmentation
        train_datagen=ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2, preprocessing_function=preprocessing_fn
            )   #apply slight zooming (+/- 0.1)

    test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocessing_fn)

    #create hsv preprocessing function
    def rgb_to_hsv(img_array):
        # 1. De-normalize and Cast to uint8 (Necessary for cv2)
        # This undoes the ImageDataGenerator's rescale=1./255.0
        # The array becomes uint8 in range [0, 255]
        img_array_uint8 = (img_array * 255).astype(np.uint8)

        # 2. Convert Color Space (HSV values are now defined by cv2: H:[0-179], S,V:[0-255])
        hsv_array = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2HSV)
        
        # 3. Cast back to float and Re-normalize (Necessary for the Model)
        # The array is now float in range [0.0, 1.0] (or similar, depending on HSV conversion)
        return hsv_array.astype(np.float32) / 255.0

    
    def two_arm_generator(generator_rgb, preprocessing_hsv):
        for batch_rgb, labels in generator_rgb:
            
            # Create the corresponding batch for HSV
            batch_hsv = np.empty_like(batch_rgb)
            
            # Apply HSV preprocessing to every image in the batch
            for i, img in enumerate(batch_rgb):
                # The preprocessing_hsv function is the one you defined before 
                # (which handles the float/uint8/normalization conversions)
                batch_hsv[i] = preprocessing_hsv(img)

            # Yield the inputs as a dictionary (mapping Input names to arrays)
            yield ({'rgb_input': batch_rgb, 'hsv_input': batch_hsv}, labels)

    #use flow_from_directory to connect to folders
    train_generator = train_datagen.flow_from_directory(    #define based on new augmented generator
        directory=f'{data_dir}/train',
        target_size=(picture_size, picture_size),
        batch_size=batch_size,
        classes=class_order,
        class_mode='binary',   # 'binary' for two classes (0 and 1)
        subset='training',
        shuffle=True
    )
    val_generator=train_datagen.flow_from_directory(   #keras ignores augmentation if subset = validation
        directory=f'{data_dir}/train',
        target_size=(picture_size, picture_size),
        batch_size=batch_size,
        classes=class_order,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    test_generator=test_datagen.flow_from_directory(  #test set not augmented
        directory=f'{data_dir}/test',
        target_size=(picture_size, picture_size),
        batch_size=batch_size,
        classes=class_order,
        class_mode='binary',
        shuffle=False
    )

    train_2_arm = two_arm_generator(train_generator, rgb_to_hsv)
    val_2_arm = two_arm_generator(val_generator, rgb_to_hsv)
    test_2_arm = two_arm_generator(test_generator, rgb_to_hsv)

    #calculate required training steps explicitly since we used custom generator
    
    train_stpes= math.ceil(train_generator.samples / batch_size)
    val_steps = math.ceil(val_generator.samples / batch_size)
    test_steps = math.ceil(test_generator.samples / batch_size)

    return train_2_arm, val_2_arm, (test_2_arm, test_generator.filenames), train_stpes, val_steps, test_steps
        
    
    
   