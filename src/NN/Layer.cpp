#include "./NN/Layer.hpp"
#include <vector>
#include <functional>
#include "./Functional/Activations.hpp"

LinearLayer::LinearLayer(int input_size, int output_size, const std::string &activation)
    : input_size(input_size), output_size(output_size), activation(activation)
{
    create_Init(); // Initialize weights and biases in the constructor
}

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
    if (this->activation == "sigmoid")
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
