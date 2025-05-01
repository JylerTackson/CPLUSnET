#include "./Functional/Loss.hpp"

// Mean Squared Error loss function
double Loss::mse(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets)
{
    // Formula: MSE = (1/n) * sum((predictions - targets)^2)
    // squaredNorm() computes the sum of squares of the vector
    return (predictions - targets).squaredNorm() / predictions.size();
}

// Cross entropy loss function
double Loss::cross_entropy(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets)
{
    // Formula: Cross-Entropy = -sum(targets * log(predictions + epsilon))
    // where epsilon is a small value to prevent log(0)
    double epsilon = 1e-15;
    return -(targets.array() * (predictions.array() + epsilon).log()).sum() / targets.size();
}

//-------------------------------------------------------------------------

// MSE derivative function
Eigen::VectorXd DerivativeLoss::mse_derivative(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets)
{
    // Formula: dMSE/dPredictions = (2/n) * (predictions - targets)
    // where n is the number of predictions
    return (2.0 / predictions.size()) * (predictions - targets);
}

// Cross-Entropy derivative function
Eigen::VectorXd DerivativeLoss::cross_entropy_derivative(const Eigen::VectorXd &predictions, const Eigen::VectorXd &targets)
{
}
