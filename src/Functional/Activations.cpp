#include "./Functional/Activations.hpp"
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
// https://eigen.tuxfamily.org/dox/namespaceEigen.html#a0110c233d357169fd58fdf5656992a98

// #include <Eigen/Dense>
// This library is used for matrix and vector operations in C++.
// You cannot perform element wise operations on Eigen::MatrixXd || Eigen::VectorXd objects directly.
// You need to use the .array() method to convert them to an array type first.
// Then you can perform element wise operations (such as exp, max, etc.) on the array type.
// Finally, Eigen is implicitlly converting the array type back to the matrix or vector type when you return it from the function.
// If you would rather do it explicity, you can use the .matrix() method to convert it back to the matrix type.

// Forwards Propagation activations
//------------------------------------------------------------------------

// Sigmoid activation function
Eigen::VectorXd ForwardActivations::sigmoidActivation(const Eigen::VectorXd &x)
{
    // sigmoid function: 1 / (1 + exp(-x))
    return 1.0 / (1.0 + (-x).array().exp());
}

// ReLU activation function
Eigen::VectorXd ForwardActivations::reluActivation(const Eigen::VectorXd &x)
{
    // ReLU function: max[0, x]
    return x.array().max(0);
}

// Tanh activation function
Eigen::VectorXd ForwardActivations::tanhActivation(const Eigen::VectorXd &x)
{
    // tanh function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    // Use Eigen's built-in tanh function for better performance
    return x.array().tanh();
}

// Softmax activation function
Eigen::VectorXd ForwardActivations::softmaxActivation(const Eigen::VectorXd &x)
{
    // Softmax function: exp(x) / sum(exp(x))
    Eigen::VectorXd exp_x = x.array().exp();
    return exp_x / exp_x.sum();
}

// Backwards Propagation activations
//------------------------------------------------------------------------

// Derivative of the sigmoid function
Eigen::VectorXd BackwardActivations::sigmoidDerivative(const Eigen::VectorXd &x)
{
    // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
    ForwardActivations forwardActivations;                         // create an instance of Forward Activations class
    Eigen::VectorXd sig = forwardActivations.sigmoidActivation(x); // call sigmoid function using instance
    return sig.array() * (1 - sig.array());
}

// Derivative of the ReLU function
Eigen::VectorXd BackwardActivations::reluDerivative(const Eigen::VectorXd &x)
{
    // Derivative of ReLU: 1 if x > 0, else 0
    // Cast to Array to perform element wise comparison and casting to double
    return (x.array() > 0).cast<double>();
}

// Derivative of the Tanh function
Eigen::VectorXd BackwardActivations::tanhDerivative(const Eigen::VectorXd &x)
{
    // Derivative of tanh: 1 - tanh^2(x)
    // Use Eigen's built-in tanh function for better performance
    return (1 - x.array().tanh().pow(2));
}

// Derivative of the Softmax function
Eigen::VectorXd BackwardActivations::softmaxDerivative(const Eigen::VectorXd &x)
{
    // Derivative of softmax: softmax(x) * (1 - softmax(x))
    // This is a bit more complex, as it involves the Jacobian matrix of the softmax function
}

// Jacobian matrix for softmaxDerivative
Eigen::MatrixXd BackwardActivations::jacobian(const Eigen::VectorXd &x)
{
    // Create a diagonal matrix with the softmax values
}
