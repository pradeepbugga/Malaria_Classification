#gradcam_utils.py
#this file contains utility functions for Grad-CAM visualization
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model



def build_feature_and_classifier(model, last_conv_name):
    """Split the model into feature extractor and classifier parts."""
    last_conv_layer = model.get_layer(last_conv_name)
    last_conv_idx = [i for i, l in enumerate(model.layers) if l.name == last_conv_name][0]

    feature_extractor = Model(inputs=model.inputs, outputs=last_conv_layer.output)

    x = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    z = x
    for layer in model.layers[last_conv_idx + 1:]:
        z = layer(z)
    classifier = Model(inputs=x, outputs=z)

    return feature_extractor, classifier

def load_tensor_rgbs(path, target_size=128):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import tensorflow as tf

    img = load_img(path, target_size=(target_size, target_size))
    rgb = img_to_array(img).astype("float32") / 255.0
    hsv = tf.image.rgb_to_hsv(rgb)
    S = hsv[:, :, 1:2]  # pick the S channel for example
    rgbh = tf.concat([rgb, S], axis=-1)
    return img, np.expand_dims(rgbh.numpy(), 0)

def compute_gradcam(feature_extractor, classifier, img_tensor):
    # Compute Grad-CAM for the given image tensor.
    with tf.GradientTape() as tape:
        A = feature_extractor(img_tensor, training=False)   # [1,H,W,C]
        tape.watch(A)
        preds = classifier(A, training=False)               # [1,1]
        y = tf.clip_by_value(preds[:,0], 1e-7, 1-1e-7)      # class 1 score
        logit = tf.math.log(y/(1-y))

    grads = tape.gradient(logit, A)                         # ∂logit/∂A
    pooled = tf.reduce_mean(grads, axis=(1,2), keepdims=True)  # [1,1,1,C]
    cam = tf.nn.relu(tf.reduce_sum(pooled * A, axis=-1))[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, float(preds[0,0])

def compute_gradcam_rgb_only(feature_extractor, classifier, img_tensor):
    """
    Compute Grad-CAM while ignoring the 4th (S) channel.
    img_tensor: (1, H, W, 4)
    """
    if isinstance(img_tensor, np.ndarray):
        img_rgb_only = img_tensor.copy()
    else:
        img_rgb_only = img_tensor.numpy().copy()

    img_rgb_only[..., 3] = 0.0
    img_rgb_only = tf.convert_to_tensor(img_rgb_only, dtype=tf.float32)

    with tf.GradientTape() as tape:
        A = feature_extractor(img_rgb_only, training=False)
        tape.watch(A)
        preds = classifier(A, training=False)
        y = tf.clip_by_value(preds[:, 0], 1e-7, 1 - 1e-7)
        logit = tf.math.log(y / (1 - y))

    grads = tape.gradient(logit, A)
    pooled = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(pooled * A, axis=-1))[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, float(preds[0, 0])

def overlay_cam(pil_img, cam):
    h, w = pil_img.size[1], pil_img.size[0]
    camr = cv2.resize(cam, (w, h))
    heat = cv2.applyColorMap((camr*255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    out  = cv2.addWeighted(base, 0.5, heat, 0.5, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)




def generate_gradcam(model_path, img_path, last_conv_name, output_path, picture_size=128):
    """Convenience wrapper for full Grad-CAM pipeline."""
    tf.config.run_functions_eagerly(True)
    tf.keras.backend.set_floatx("float32")

    model = load_model(model_path)
    _ = model(tf.zeros((1, picture_size, picture_size, 4), tf.float32))

    feature_extractor, classifier = build_feature_and_classifier(model, last_conv_name)
    img, x = load_tensor_rgbs(img_path, picture_size)
    cam, prob = compute_gradcam_rgb_only(feature_extractor, classifier, x)
    overlay = overlay_cam(img, cam)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return prob

def saliency_rgb_only(model, img_tensor):
    """Compute RGB-only input saliency (gradient magnitude)."""
    import tensorflow as tf, numpy as np, cv2

    if isinstance(img_tensor, np.ndarray):
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor, training=False)
        y = tf.math.log(preds[:, 0] / (1 - preds[:, 0]))

    grads_input = tape.gradient(y, img_tensor)[0].numpy()
    rgb_grad = grads_input[..., :3]
    grad_mag = np.mean(np.abs(rgb_grad), axis=-1)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

    rgb_img = img_tensor[0, ..., :3].numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

    heat = cv2.applyColorMap((grad_mag * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = (rgb_img * 255).astype(np.uint8) 
    over = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.5, heat, 0.5, 0)
    return cv2.cvtColor(over, cv2.COLOR_BGR2RGB)

def saliency_rgb_only_sidebyside(model, img_tensor, save_path=None):
    """
    Compute RGB-only input saliency (gradient magnitude),
    overlay it on the RGB image, and place both side by side.
    """
    import tensorflow as tf, numpy as np, cv2, os

    # Ensure tensor
    if isinstance(img_tensor, np.ndarray):
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    # ---- Compute gradient wrt input ----
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor, training=False)
        y = tf.math.log(preds[:, 0] / (1 - preds[:, 0]))

    grads_input = tape.gradient(y, img_tensor)[0].numpy()  # (H,W,4)
    rgb_grad = grads_input[..., :3]
    grad_mag = np.mean(np.abs(rgb_grad), axis=-1)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

    # ---- Normalize and prepare RGB base image ----
    rgb_img = img_tensor[0, ..., :3].numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    base = (rgb_img * 255).astype(np.uint8)

    # ---- Create heatmap overlay ----
    heat = cv2.applyColorMap((grad_mag * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.5, heat, 0.5, 0)
    over_rgb = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)

    # ---- Combine side by side ----
    combined = np.hstack([base, over_rgb])

    # ---- Save if requested ----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    return combined