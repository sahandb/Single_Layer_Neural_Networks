#  Single Layer Perceptron Discrete and Continuous

Implementation of discrete and continuous single-layer neuron perceptron neural network

At first I read the all images from train and test and somepics folder then turn them from RGB to gray and binary(for discrete: binary with 1 and -1 intensities and for gray 0 to 255)
Resize all photos to 622 * 534 px and create noisy(s&p) data for train test and somepic with 20%
Fix the labels and turn them to 0 to 5 for hands in all folders
Then flatten images from matrix to vector for feeding perceptron
And then we have 2 types of activation function one for discrete(unit step) and the other continuous(sigmoid) 
I use pre defined function in sklearn library for single layer perceptron in python and it use activation function whenever need to and don’t need any configuration for that

I use 0.15 for learning rate of perceptron and feed perceptron with random vector for better performance and at the end get accuracies 
I comment all of description in my code too 


# How Training Perceptron work?

Training Continuous Perceptron
Activation function: Sigmoid
We stop the loop when the changes in weights are lower than threshold


Training Discrete Perceptron 
Activation function: Unit step
We stop the loop when we have zero changes in weight

![image](https://user-images.githubusercontent.com/24508376/219617932-7fc6e9e9-d36c-446b-94c8-7e8bbe86e985.png)
