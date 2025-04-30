#ifndef LAYER_HPP
#define LAYER_HPP
#include "./Base/TyNET.hpp"
#include <array>
#include <functional>

// This class will create a layer of the network.

class LinearLayer : public TyNET
{

public:
    // Constructor to initialize the layer with input and output sizes
    // Good to pass a const reference to the constructor to avoid copying
    LinearLayer(int input_size, int output_size, const std::array<char, 100> &activation);

    // Create & initialize weights matrix and bias vector
    void create_Init(int input_size, int output_size); // Function to create weights and biases



    ~LinearLayer(); // Destructor

private:
    // Class Attributes:
    // The following attributes are used to define the layer

    //Initialization of the layer
    int input_size;                   // Size of the input to the layer
    int output_size;                  // Size of the output from the layer

    std::array<char, 100> activation; // Activation function to be used in the layer

    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_function;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_derivative;

    Eigen::MatrixXd weights;         // Weights of the layer
    Eigen::VectorXd biases;         // Biases of the layer
};

#endif