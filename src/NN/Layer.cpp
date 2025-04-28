#include "./NN/Layer.hpp"
#include <vector>
#include <functional>
#include "./Functional/Activations.hpp"

LinearLayer::LinearLayer(int input_size, int output_size, const std::array<char, 100> &activation)
    : input_size(input_size), output_size(output_size), activation(activation)
{
    initialize_weights_and_biases(); // Initialize weights and biases in the constructor
}

void LinearLayer::initialize_weights_and_biases()
{
    // Initialize weights and biases using a normal distribution
    this->weights = Eigen::MatrixXd::Random(output_size, input_size); // Random weights
    this->biases = Eigen::VectorXd::Zero(output_size);                // Zero biases
}