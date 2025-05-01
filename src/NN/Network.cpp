#include "./NN/Network.hpp"
#include "./Functional/Activations.hpp"
#include "./Functional/BackProp.hpp"
#include "./Functional/ForwardProp.hpp"
#include <vector>
#include <iostream>
#include <cmath>

Network::Network(const std::vector<LinearLayer> &layers, double learning_rate, double momentum)
    : layers(layers), learning_rate(learning_rate), momentum(momentum)
{
    network_Layers = layers.size(); // Initialize the size of the network
    // TODO: Catches for invalid network sizes
    // TODO: Catches for invalid learning rates and momentum values
}

void Network::train(const Eigen::VectorXd &input,
                    const Eigen::VectorXd &target,
                    double learning_rate,
                    double momentum,
                    int epochs,
                    int batch_size)
{
    Eigen::VectorXd output; // Placeholder for the predicted output vector

    // TODO: Implement the training loop
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
    }
}

void Network::predict(const Eigen::VectorXd &input)
{
    // Call the forward method to get the output from the network
    forward(input);
    // print output
    std::cout << "Predicted Output: " << std::endl;
}

void Network::forward(const Eigen::VectorXd &input)
{
    // TODO: Implement the forward propagation through the network
    ForwardProp fp;
}

void Network::backward(const Eigen::VectorXd &output)
{
    // TODO: Implement the backward propagation through the network
    BackProp bp;
}