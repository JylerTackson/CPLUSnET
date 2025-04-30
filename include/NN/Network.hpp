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
    Network();
    ~Network();

    // Function to create the network from the layers using a vector
    void create_network(const std::vector<LinearLayer> &layers);

private:

    // Class Attributes:
    
};

#endif