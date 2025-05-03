// PyTorch Linear.py:
// https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py

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
        void create_Init();

        ~LinearLayer(); // Destructor

    private:
        // Class Attributes:
        //----------------------------------------------------------------------------

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



        //Private Methods:
        //----------------------------------------------------------------------------
        // Function to perform forward propagation through the layer

        Eigen::VectorXd forward(const Eigen::VectorXd &input);


        // Function to perform backward propagation through the layer
        // This method should most likely be built into the training method as we have to do 
        // interim calculations to get the gradients and update the weights & biases.
        void train();

        
        void updateWeights(const Eigen::MatrixXd &dL_dw); // Function to update weights


};

#endif