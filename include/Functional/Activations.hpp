#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <cmath>
#include <iostream>
#include <Eigen/Dense>

// VectorXd:
// X -> 'dynamic size' set at runtime
// d -> double precision floats

class ForwardActivations : public TyNET
{
public:
    // Activation function for Eigen::VectorXd inputs
    Eigen::VectorXd sigmoid(const Eigen::VectorXd &x); // Sigmoid activation function
    Eigen::VectorXd relu(const Eigen::VectorXd &x);    // ReLU activation function
    Eigen::VectorXd tanh(const Eigen::VectorXd &x);    // Tanh activation function
    Eigen::VectorXd softmax(const Eigen::VectorXd &x); // Softmax activation function

private:
};

class BackwardActivations
{
public:
    // Derivative of the activation functions
    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd &x); // Derivative of the sigmoid function
    Eigen::VectorXd relu_derivative(const Eigen::VectorXd &x);    // Derivative of the ReLU function
    Eigen::VectorXd tanh_derivative(const Eigen::VectorXd &x);    // Derivative of the Tanh function
    Eigen::VectorXd softmax_derivative(const Eigen::VectorXd &x); // Derivative of the Softmax function

private:
    Eigen::MatrixXd jacobian(const Eigen::VectorXd &x); // Jacobian matrix for softmaxDerivative
};

#endif
