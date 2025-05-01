#ifndef LAYER_HPP
#define LAYER_HPP
#include "./Base/TyNET.hpp"
#include <array>
#include <string>
#include <functional>

// This class will create a layer of the network.

class LinearLayer : public TyNET
{

public:
    // Constructor to initialize the layer with input and output sizes
    // Good to pass a const reference to the constructor to avoid copying
    LinearLayer(int input_size, int output_size, const std::string &activation);

    // Create & initialize weights matrix and bias vector
    void create_Init(); // Function to create weights and biases

    ~LinearLayer(); // Destructor

private:
    // Class Attributes:
    // The following attributes are used to define the layer

    // Initialization of the layer
    int input_size;         // Size of the input to the layer
    int output_size;        // Size of the output from the layer
    std::string activation; // Activation function to be used in the layer

    // Activation function and its derivative
    // These are function pointers to the activation function and its derivative
    // std::function is a C++ standard library feature that allows you to store callable objects
    // I am defining the function that it is pointing to as a function that takes an Eigen::VectorXd and returns an Eigen::VectorXd
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> activation_function;
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> activation_derivative;

    // Weights and biases
    Eigen::MatrixXd weights; // Weights between the input and output of the layer
    Eigen::VectorXd biases;  // Biases of the output layer

    // Layer Values
    Eigen::VectorXd pre_Activated; // Pre-activated values of the output layer
    Eigen::VectorXd activated;     // Activated values of the output layer
};

#endif