# NatHacks22NeuralSignalDecoding

This is a project for NatHacks2022 individual submission. 

The goal of this project is using PyTorch to process mouse brain neuron signal data in order 
to train a model that automatically infers which type of sound is heard by the subject when 
an external sound is being played to the subject (where the sound has to be one of the 10 
sounds in the training dataset). The frameworks or libraries used in the project include 
PyTorch, NumPy and Scipy, and Matlab is used to save, load and visualize the data.


### Model Structure

The model used to learn the classification of sound being played is simple, consisting of a 
pooling layer to reduce the input size (as in the provided dataset each trial has 85 neurons 
recorded for 251 time steps) and a linear regression classifier. The data for training and 
testing will be loaded to GPU memory and processed there if you have a NVIDIA GPU available.
The model obtains 94% accuracy on reserved test set of 200 trials.


### Usage

To run the program, you need to have Python environment setup for [PyTorch](https://pytorch.org/).
You can learn a model and test it using main.py as reference, which takes a .mat file and split it 
to training/testing datasets and train/test the model.


### Acknowledgements

I would acknowledge that the [PyTorch webpage tutorial](https://pytorch.org/tutorials/) has 
helped me a lot during writing this program, as I've only picked up this as my first ML library 
and learned most of the implementation from the tutorial.

I would also like to thank NatHacks for hosting workshops on basics of ML, including introduction 
to techniques used for supervised/unsupervised learning, which has inspired me to try out a 
few other approaches to this problem besides linear regression.