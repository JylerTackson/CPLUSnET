#ifndef BACKPROP_HPP
#define BACKPROP_HPP
#include "./Base/TyNET.hpp"
#include <Eigen/Dense>

class BackProp : public TyNET
{
public:
    BackProp();

    ~BackProp();

private:
    // Sets the sizes for the gradientVectors
    void size();

    // Calculates the gradients for the weights and biases of the layer
    void calculateGradients();

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
};

#endif