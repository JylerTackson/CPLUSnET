#include "./NN/Layer.hpp"
#include <vector>
#include <functional>
#include "./Functional/Activations.hpp"

LinearLayer::LinearLayer(int input_size, int output_size, const std::array<char, 100> &activation)
    : input_size(input_size), output_size(output_size), activation(activation)
{
    initialize_weights_and_biases(input_size, output_size, activation); // Initialize weights and biases in the constructor
}


void LinearLayer::create_Init(int input_size, int output_size)
{
    // Initialize weights and biases using the specified sizes
    this->weights = Eigen::MatrixXd::Random(output_size, input_size);   // Random weights [-1, 1]
    this->weights = this->weights * 0.1;                                // Scale weights to [-0.1, 0.1]
    this->biases = Eigen::VectorXd::Zero(output_size);                  // Zero biases

    ForwardActivations forward;
    BackwardActivations backward;

    // Set activation function based on the provided string
    if (activation == "sigmoid")
    {
        this->activation_function = std::bind(&ForwardActivations::sigmoidActivation, &forward, std::placeholders::_1);
        this->activation_derivative = std::bind(&BackwardActivations::sigmoidDerivative, &backward, std::placeholders::_1);
    }
    else if (activation == "relu")
    {
        this->activation_function = std::bind(&ForwardActivations::reluActivation, &forward, std::placeholders::_1);
        this->activation_derivative = std::bind(&BackwardActivations::reluDerivative, &backward, std::placeholders::_1);
    }
    else if (activation == "tanh")
    {
        this->activation_function = std::bind(&ForwardActivations::tanhActivation, &forward, std::placeholders::_1);
        this->activation_derivative = std::bind(&BackwardActivations::tanhDerivative, &backward, std::placeholders::_1);
    }
    else if (activation == "softmax")
    {
        this->activation_function = std::bind(&ForwardActivations::softmaxActivation, &forward, std::placeholders::_1);
        this->activation_derivative = std::bind(&BackwardActivations::softmaxDerivative, &backward, std::placeholders::_1);
    }

}

