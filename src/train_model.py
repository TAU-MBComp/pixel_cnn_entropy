#!/usr/bin/env python
""" Train the model """
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
from tensorflow_probability import distributions as tfd
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from utils import read_config, load_input_data, build_model

# Read the configuration file
c = read_config("run.param")

# Initialize dataset and split into training and validation sets
image_shape = (c.L, c.L, 1)
data = load_input_data(c.filename, c.L, image_shape)
train_data = data.take(c.training_samples)
validation_data = data.skip(c.training_samples).take(c.test_samples)


def image_preprocess(x):
    """Cast image pixel values to float32"""
    x['image'] = tf.cast(x['image'], tf.float32)
    return (x['image'], )


train_it = train_data.map(image_preprocess).batch(c.batch_size).shuffle(
    c.shuffbuff)
validation_it = validation_data.map(image_preprocess).batch(
    c.batch_size).shuffle(c.shuffbuff)

# Define a model
model, dist_ = build_model(image_shape,
                           c.num_resnet,
                           c.hierarchies,
                           c.filters,
                           c.logistic_mix,
                           c.dropout,
                           high=1)

# Compile and train the model
model.compile(optimizer=tfk.optimizers.Adam(c.learning_rate), metrics=[])

# Save weights to  checkpoint file
checkpoint_path = "weights_{epoch:d}/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_freq='epoch',
                                                 save_best_only=False,
                                                 save_weights_only=True,
                                                 verbose=1)
print("Fitting...")
H = model.fit(train_it,
              epochs=c.epochs,
              verbose=2,
              validation_data=validation_it,
              callbacks=[cp_callback])  # Pass callback to training

# Save training loss and validation loss
np.savetxt('loss.dat', H.history["loss"])
np.savetxt('validation_loss.dat', H.history["val_loss"])
