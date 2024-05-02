"""Utility functions for the PixelCNN++ entropy method."""
import configparser
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from dataclasses import dataclass
import numpy as np

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers


@dataclass
class Config:
    """Parameters for the model and entropy calculation."""
    filename: str
    T: float
    L: int
    epochs: int
    num_resnet: int
    batch_size: int
    learning_rate: float
    shuffbuff: int
    training_samples: int
    test_samples: int
    hierarchies: int
    filters: int
    logistic_mix: int
    dropout: float
    entropy_samples: int


def read_config(filename):
    """Read the configuration file."""
    config = configparser.ConfigParser()
    config.read(filename)
    filename = str(config['input']['filename'])
    T = float(config['input']['T'])
    L = int(config['input']['L'])
    epochs = int(config['input']['epochs'])
    num_resnet = int(config['input']['num_resnet'])
    batch_size = int(config['input']['batch_size'])
    learning_rate = float(config['input']['learning_rate'])
    shuffbuff = int(config['input']['shuffbuff'])
    training_samples = int(config['input']['training_samples'])
    test_samples = int(config['input']['test_samples'])
    hierarchies = int(config['input']['hierarchies'])
    filters = int(config['input']['filters'])
    logistic_mix = int(config['input']['logistic_mix'])
    dropout = float(config['input']['dropout'])
    entropy_samples = int(config['input']['entropy_samples'])
    c = Config(filename, T, L, epochs, num_resnet, batch_size, learning_rate,
               shuffbuff, training_samples, test_samples, hierarchies, filters,
               logistic_mix, dropout, entropy_samples)
    return c


def load_input_data(filename, L, shape, buffer_size=1):
    """Load the input data from a '.npz' file."""

    def sample_prep(sample):
        sample = np.array(sample)
        sample = np.where(sample == -1, 0, sample)
        sample = sample[..., np.newaxis]
        sample_dict = {'image': sample, 'label': ()}
        return sample_dict

    def get_samples(filename):
        samples = np.load(filename)
        for sname in samples.files:
            yield sample_prep(samples[sname])

    ds_samples = tf.data.Dataset.from_generator(get_samples,
                                                args=[filename],
                                                output_types={
                                                    'image': tf.int64,
                                                    'label': tf.float64
                                                },
                                                output_shapes={
                                                    'image': shape,
                                                    'label': None
                                                })
    ds_samples = tf.data.Dataset.prefetch(ds_samples, buffer_size)
    return ds_samples


def build_model(image_shape, num_resnet, hierarchies, filters, logistic_mix,
                dropout, high):
    """Build a PixelCNN++ model."""
    # Define a Pixel CNN network
    dist = tfd.PixelCNN(image_shape=image_shape,
                        num_resnet=num_resnet,
                        num_hierarchies=hierarchies,
                        num_filters=filters,
                        num_logistic_mix=logistic_mix,
                        dropout_p=dropout,
                        high=high)

    # Define the model input
    image_input = tfkl.Input(shape=image_shape)

    # Define the log likelihood for the loss fn
    log_prob = dist.log_prob(image_input)

    # Define the model
    model = tfk.Model(inputs=image_input, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))
    return model, dist
