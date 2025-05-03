// PyTorch Container.py:
// https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/container.py

// What I am trying to achieve with this Network class was the "Sequential" class in PyTorch
// This class is a container for layers and allows you to add layers to the network in a sequential manner

#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "./Base/TyNET.hpp"
#include "./NN/Layer.hpp"
#include <vector>

// This class will receieve an array of layers and create a network from them
// Why use vectors instead of arrays?
// Vectors are dynamic in size and can be resized at runtime, while arrays have a fixed size
// Because of this, we can do things such as Dropout, Batch Normalization, and other dynamic operations
// that require the size of the network to be changed at runtime

class Network : public TyNET
{
    public:
        // Constructor to initialize the network from a vector of layers
        // Simply initialize the layers vector, layers size, learning rate, and momentum.
        Network(const std::vector<LinearLayer> &layers, double learning_rate, double momentum);

        // Function to train the network
        // Utilizes the forward and backward methods to train the network
        void train(const Eigen::VectorXd &input,
                const Eigen::VectorXd &target,
                double learning_rate,
                double momentum,
                int epochs,
                int batch_size);

        // Function that uses the trained network to predict an output from an input
        // Input is a vector of size input_size
        void predict(const Eigen::VectorXd &input);

        ~Network();

    private:
        // Forward Propagation:
        // This method will take an input and pass it through the network
        // Needs to create instance of ForwardProp.hpp
        void forward(const Eigen::VectorXd &input);

        // Backward Propagation:
        // This method will take an output and pass it through the network by propogating it backwards
        // Needs to create instance of BackwardProp.hpp
        void backward(const Eigen::VectorXd &output);

        // Class Attributes:
        int network_Layers;              // Size of the network
        std::vector<LinearLayer> layers; // Array of layers in the network

        double learning_rate; // Learning rate for the network
        double momentum;      // Momentum for the network

        // Gradient Variables:
        // Gradient of the loss function with respect to the output of the layer
        Eigen::VectorXd dL_dy;
        // Gradient of the loss function with respect to the pre-activated values of the layer
        Eigen::VectorXd dL_dz;
        // Gradient of the loss function with respect to the post-activated values of layer k-1
        Eigen::VectorXd dL_dyk_1;
        // Gradient of the loss function with respect to the weights of the layer
        Eigen::MatrixXd dL_dw;
        // Gradient of the loss function with respect to the biases of the layer
        Eigen::VectorXd dL_db;

        // Information available in Layer Class:
        //  - input_size
        //  - output_size
        //  - activationFunction
        //  - activationDerivative
        //  - weightsMatrix
        //  - biases
        //  - pre_Activated values of OUTPUT layer
        //  - activated values of OUTPUT layer


};

#endif