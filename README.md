# Density estimation approach to evaluating entropy

## General Information
This is a simple implementation of the density estimation algorithm for entropy estimation, based on the PixelCNN model.
Note that this uses the implementation of PixelCNN++ found in the `tensorflow-probability` library, which requires very particular versions of python, TensorFlow, and other dependencies.
To make it easier to run reproducibly, we provide a Dockerfile that sets up the environment for you.
To run the example with Docker, follow the instructions below.
For substantially larger applications, we recommend using a more modern implementation and training on a GPU.

## Running the example
This will perform a calculation using the data provided in `data/ising_samples.npz`, which are 3000 configurations from an equilibrium 2D Ising model on a square lattice of size $L=16$ and at a temperature of $T=2.5$.
Parameters for the calculation are provided in `example/run.param`, which specifies the number of epochs to train for, the batch size, and the learning rate among other things.
Assuming you have Docker and git installed, you can run the following commands to clone the repository and build the Docker image:
```bash
git clone git@github.com:TAU-MBComp/pixel_cnn_entropy.git
cd pixel_cnn_entropy
docker build -t pcnne .
```
To learn the density from the simulated data, run the following command:
```bash
docker run -v $PWD:/data -w /data/example -it pcnne python3 ../src/train_model.py
```
This will save the trained weights and learning curves on disk.
To use the weights to predict entropy and compare the result to the exact solution, run the following command:
```bash
docker run -v $PWD:/data -w /data/example -it pcnne python3 ../src/predict_entropy.py
```
