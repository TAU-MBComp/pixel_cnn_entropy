#!/usr/bin/env python
""" Predict the entropy of the trained model """
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from utils import read_config, build_model, load_input_data
from exact_solution import entropy

# Read the configuration file
c = read_config("run.param")
image_shape = (c.L, c.L, 1)

# Use existing samples to avoid the need for generation
data = load_input_data(c.filename, c.L, image_shape)
data = data.shuffle(c.shuffbuff)
samples_mc = data.take(c.entropy_samples)

# Load the trained model
model, dist = build_model(image_shape,
                          c.num_resnet,
                          c.hierarchies,
                          c.filters,
                          c.logistic_mix,
                          c.dropout,
                          high=1)
model.load_weights("weights_{0}/cp.ckpt".format(c.epochs))

entropies = []

for sample in samples_mc:
    entropies.append(-dist.log_prob(sample['image'].numpy()).numpy() / c.L**2)

mean_entropy = sum(entropies) / c.entropy_samples
stderr_entropy = sum([(mean_entropy - e)**2
                      for e in entropies]) / (c.entropy_samples - 1)
print(f"Estimated lower bound for entropy: {mean_entropy} ± {stderr_entropy}")
print(f"Exact entropy: {entropy(c.T, c.L)}")
