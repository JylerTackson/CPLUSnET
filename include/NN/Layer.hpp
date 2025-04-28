#ifndef LAYER_HPP
#define LAYER_HPP
#include "./Base/TyNET.hpp"
#include <array>

// This class will create a layer of the network.

class LinearLayer : public TyNET
{

public:
    // Constructor to initialize the layer with input and output sizes
    // Good to pass a const reference to the constructor to avoid copying
    LinearLayer(int input_size, int output_size, const std::array<char, 100> &activation);

    // initialize the weights and biases of the layer
    void initialize_weights_and_biases(); // Function to initialize weights and biases

    ~LinearLayer(); // Destructor

private:
    // Class Attributes:
    // The following attributes are used to define the layer

    int input_size;                   // Size of the input to the layer
    int output_size;                  // Size of the output from the layer
    std::array<char, 100> activation; // Activation function for the layer

    Eigen::MatrixXd weights; // Weights of the layer
    Eigen::VectorXd biases;  // Biases of the layer
    Eigen::VectorXd output;  // Output of the layer
};

#endif