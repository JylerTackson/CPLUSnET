//Pytorch is built with C++14 standard version.
//PyTorch uses Python as its primary interface, but it is built on top of C++ for performance reasons.
//The core libraries and many of the underlying operations are implemented in C++ for efficiency.
//The Python interface provides a more user-friendly way to interact with the library, while the C++ backend handles the heavy lifting.

//PyTorch Activation Module:
//https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py

//PyTorch Activation.cpp:
//https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp

#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "./Base/TyNET.hpp"

// VectorXd:
// X -> 'dynamic size' set at runtime
// d -> double precision floats

class ForwardActivations : public TyNET
{
public:
    // Activation function for Eigen::VectorXd inputs
    Eigen::VectorXd sigmoidActivation(const Eigen::VectorXd &x); // Sigmoid activation function
    Eigen::VectorXd reluActivation(const Eigen::VectorXd &x);    // ReLU activation function
    Eigen::VectorXd tanhActivation(const Eigen::VectorXd &x);    // Tanh activation function
    Eigen::VectorXd softmaxActivation(const Eigen::VectorXd &x); // Softmax activation function

private:
};

class BackwardActivations
{
public:
    // Derivative of the activation functions
    Eigen::VectorXd sigmoidDerivative(const Eigen::VectorXd &x); // Derivative of the sigmoid function
    Eigen::VectorXd reluDerivative(const Eigen::VectorXd &x);    // Derivative of the ReLU function
    Eigen::VectorXd tanhDerivative(const Eigen::VectorXd &x);    // Derivative of the Tanh function
    Eigen::VectorXd softmaxDerivative(const Eigen::VectorXd &x); // Derivative of the Softmax function

private:
    Eigen::MatrixXd jacobian(const Eigen::VectorXd &x); // Jacobian matrix for softmaxDerivative
};

#endif
