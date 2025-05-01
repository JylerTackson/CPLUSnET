#include <iostream>
#include "include/Base/TyNET.hpp"
#include "include/NN/Layer.hpp"
#include "include/NN/Network.hpp"

int main()
{
    // TyNET will connect the networks and allow them to communicate with each other

    Network net1({LinearLayer(3, 6, "sigmoid"),
                  LinearLayer(6, 9, "sigmoid"),
                  LinearLayer(9, 6, "sigmoid"),
                  LinearLayer(6, 3, "sigmoid")}); // Create a network with four layers

    Network net2({LinearLayer(3, 4, "relu"),
                  LinearLayer(4, 8, "relu"),
                  LinearLayer(8, 4, "relu"),
                  LinearLayer(4, 2, "relu")}); // Create a network with four layers
}