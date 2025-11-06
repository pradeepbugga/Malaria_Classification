#train_utils.py
#this file contains utility functions for training models, such as data loading and preprocessing

import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def train_and_save(model, train_gen, val_gen, test_gen, filenames, output_dir, learning_rate =0.001, use_EarlyStopping = False, use_ReduceLROnPlateau = False,
                    es_patience=3, rlrop_patience=3, epochs=20, steps_per_epoch=None, validation_steps=None, test_steps=None):
    os.makedirs(output_dir, exist_ok=True)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    callbacks = []
    if use_EarlyStopping:
        callbacks.append(
            EarlyStopping(monitor='val_loss', patience=es_patience)
        )
    if use_ReduceLROnPlateau:
        callbacks.append(
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1, cooldown=0)
            )
    callbacks.append(
            ModelCheckpoint(os.path.join(output_dir,'model.keras'), monitor='val_loss', save_best_only=True))

    history=model.fit(train_gen, validation_data=val_gen, epochs=epochs, steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, callbacks=callbacks, verbose=1)

    #save history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history.history, f)

    y_pred = model.predict(test_gen, steps=test_steps)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Handle both ImageDataGenerator and tf.data.Dataset / custom generators
    if hasattr(test_gen, "classes"):
        # Keras ImageDataGenerator
        y_true = test_gen.classes
    else:
        # tf.data or custom generator
        y_true_list = []
        for _, labels in test_gen:
            if hasattr(labels, "numpy"):
                labels = labels.numpy()      # TensorFlow tensor case
            y_true_list.append(labels)
        y_true = np.concatenate(y_true_list)

    report = classification_report(y_true, y_pred_classes, output_dict=True)
    report_rounded = {
    str(k): {m: round(v, 2) for m, v in metrics.items()}
    for k, metrics in report.items()
}
    with open(os.path.join(output_dir, "reports", "classification_report.json"), "w") as f:
        json.dump(report_rounded, f, indent=2)


    np.save(os.path.join(output_dir, "reports",  "confusion_matrix.npy"), confusion_matrix(y_true, y_pred_classes))

    # make table
    df = pd.DataFrame({
        'filename': filenames,
        'y_true': y_true.flatten(),
        'y_pred': y_pred_classes.flatten(),
        'y_prob': y_pred.flatten()
    })

    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

