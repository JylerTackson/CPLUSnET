// File: Layer.cpp
// inlcude the header file for Layer HeaderFile
#include "./NN/Layer.hpp"
//include vector header for using standard vector class
#include <vector>
//include exception header for exception handling
#include <stdexcept>
//include string header for using standard string class
#include <string>
//include Eigen library for matrix operations
#include <Eigen/Dense>
//include Activations header for activation functions
#include "./Functional/Activations.hpp"

//Constructor for LinearLayer class
LinearLayer::LinearLayer(int input_size, int output_size, const std::string &activation)
    : input_size(input_size), output_size(output_size), activation(activation)
{
    create_Init(); // Initialize weights and biases in the constructor
}

//Create and Initialize weights matrix and biases vector using input and output sizes
void LinearLayer::create_Init()
{
    // Initialize weights and biases using the specified sizes
    this->weights = Eigen::MatrixXd::Random(this->output_size, this->input_size) * 0.1; // Random weights [-0.1, 0.1]
    this->biases = Eigen::VectorXd::Zero(this->output_size);                            // Zero biases

    this->pre_Activated = Eigen::VectorXd::Zero(this->output_size); // Zero pre-activated values
    this->activated = Eigen::VectorXd::Zero(this->output_size);     // Zero activated values

    ForwardActivations forward;
    BackwardActivations backward;

    // Set activation function based on the provided string
    // Using std::bind to bind the member functions of ForwardActivations and BackwardActivations classes
    switch (this->activation)
    {
    case "relu":
        this->forward = std::bind(&ForwardActivations::ReLU, forward, std::placeholders::_1);
        this->backward = std::bind(&BackwardActivations::ReLU, backward, std::placeholders::_1);
        break;
    case "sigmoid":
        this->forward = std::bind(&ForwardActivations::Sigmoid, forward, std::placeholders::_1);
        this->backward = std::bind(&BackwardActivations::Sigmoid, backward, std::placeholders::_1);
        break;
    case "tanh":
        this->forward = std::bind(&ForwardActivations::Tanh, forward, std::placeholders::_1);
        this->backward = std::bind(&BackwardActivations::Tanh, backward, std::placeholders::_1);
        break;
    case "softmax":
        this->forward = std::bind(&ForwardActivations::Softmax, forward, std::placeholders::_1);
        this->backward = std::bind(&BackwardActivations::Softmax, backward, std::placeholders::_1);
        break;
    default:
        //std::exception is the class of all standard exceptions in C++.
        //std::logic_error is a subclass of std::exception that indicates a logic error in the program.
        //std::invalid_argument is a subclass of std::logic_error that indicates an invalid argument was passed to a function.
        //https://en.cppreference.com/w/cpp/error/invalid_argument
        throw std::invalid_argument("Invalid activation function: " + this->activation);
    }
}
