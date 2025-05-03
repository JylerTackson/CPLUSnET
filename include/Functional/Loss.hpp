// Please review ref.md for more information.

// PyTorch Loss Module:
// https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py

// PyTorch Loss.cpp:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/loss.cpp

#ifndef LOSS_HPP
#define LOSS_HPP
#include "./Base/TyNET.hpp"
#include <Eigen/Dense>
#include <cmath>

class Loss : public TyNET
{
public:
    // Loss functions
    double mse(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets);           // Mean Squared Error loss
    double cross_entropy(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets); // Cross-Entropy loss

private:
};

// Derivative of the loss functions for Backpropagation

class DerivativeLoss : public TyNET
{
public:
    // Derivative of the loss functions
    Eigen::VectorXd mse_derivative(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets);           // Derivative of MSE loss
    Eigen::VectorXd cross_entropy_derivative(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets); // Derivative of Cross-Entropy loss

private:
};

#endif // DerivativeLoss
// End of file
