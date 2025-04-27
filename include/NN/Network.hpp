#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "./Base/TyNET.hpp"

//This class will receieve an array of layers and create a network from them


class Network : public TyNET
{
public:
    Network();
    ~Network();

    //Training the network with the given data and labels
    void train(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels);

    //Predicting the output for the given input data
    Eigen::MatrixXd predict(const Eigen::MatrixXd &input_data);

private:

};


#endif