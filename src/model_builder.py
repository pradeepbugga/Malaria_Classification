#model_builder.py
#This script contains functions to build different CNN model architectures for image classification

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, BatchNormalization, GlobalAveragePooling2D, Input, Concatenate

def cnn_rgb_1l(picture_size=128):
    model=Sequential()

    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation='relu', input_shape=(picture_size,picture_size,3)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))  #final classification
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid')) #last layer (1 sigmoid that will result in a value between 0 and 1 corresponding to either of two classes)

    return model

def cnn_rgb_3l(picture_size=128):
    model=Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation=None, input_shape=(picture_size,picture_size,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model    



def cnn_rgbh_3l(picture_size=128):

    model=Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation=None, input_shape=(picture_size,picture_size,4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model 


def cnn_rgbh_5l(picture_size=128):

    model=Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation=None, input_shape=(picture_size,picture_size,4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model    

def cnn_rgbs_5l(picture_size=128):
    return cnn_rgbh_5l(picture_size=picture_size)


def create_feature_extractor(input_tensor, name_suffix):
    """Defines the shared convolutional feature extraction layers (based on cnn_model4)."""
    
    # Block 1
    x = Conv2D(filters=32, kernel_size=3, padding="same", activation=None, name=f'conv1_{name_suffix}')(input_tensor)
    x = BatchNormalization(name=f'bn1_{name_suffix}')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=2, name=f'pool1_{name_suffix}')(x)

    # Block 2
    x = Conv2D(filters=64, kernel_size=3, padding="same", activation=None, name=f'conv2_{name_suffix}')(x)
    x = BatchNormalization(name=f'bn2_{name_suffix}')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=2, name=f'pool2_{name_suffix}')(x)

    # Block 3
    x = Conv2D(filters=128, kernel_size=3, padding="same", activation=None, name=f'conv3_{name_suffix}')(x)
    x = BatchNormalization(name=f'bn3_{name_suffix}')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=2, name=f'pool3_{name_suffix}')(x)
    
    # Global Pooling
    x = GlobalAveragePooling2D(name=f'gap_{name_suffix}')(x)
    
    return x

def cnn_rgb_hsv_dual_3l(picture_size=128):
    # --- 1. Define Two Inputs (Must match generator output keys) ---
    rgb_input = Input(shape=(picture_size, picture_size, 3), name='rgb_input')
    hsv_input = Input(shape=(picture_size, picture_size, 3), name='hsv_input')

    # --- 2. Feature Extraction ---
    # Process RGB input
    rgb_features = create_feature_extractor(rgb_input, name_suffix='rgb')
    # Optional: Refinement layer before fusion
    rgb_features = Dense(64, activation='relu', name='rgb_latent')(rgb_features) 
    
    # Process HSV input (using the same structure)
    hsv_features = create_feature_extractor(hsv_input, name_suffix='hsv')
    # Optional: Refinement layer before fusion
    hsv_features = Dense(64, activation='relu', name='hsv_latent')(hsv_features) 

    # --- 3. Feature Fusion ---
    # Concatenate the latent feature vectors
    merged_features = Concatenate(name='fusion_layer')([rgb_features, hsv_features])
    
    # --- 4. Classification Head (The final layers from cnn_model4) ---
    # Dense layer 1
    x = Dense(32, activation='relu')(merged_features)
    
    # Dropout
    x = Dropout(0.4)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(x)

    # --- 5. Create the Final Model ---
    model = Model(
        inputs=[rgb_input, hsv_input],
        outputs=output,
        name='Two-Arm_RGB_HSV_CNN'
    )
    
    return model

# --- Factory function for convenience ---
def get_model(name, **kwargs):
    builders = {
        "cnn_rgb_1l": cnn_rgb_3l,
        "cnn_rgb_3l": cnn_rgb_3l,
        "cnn_rgbh_3l": cnn_rgbh_3l,
        "cnn_rgbh_5l": cnn_rgbh_5l,
        "cnn_rgbs_5l": cnn_rgbs_5l,
        "cnn_rgb_hsv_dual_3l": cnn_rgb_hsv_dual_3l
        }
    if name not in builders:
        raise ValueError(f"Unknown model name: {name}")
    return builders[name](**kwargs)