#ifndef LAYER_HPP
#define LAYER_HPP
#include "./Base/TyNET.hpp"

// This class will create a layer of the network.

class LinearLayer : public TyNET
{

public:
    LinearLayer(int input_size, int output_size); // Constructor to initialize the layer with input and output sizes
    
    //initialize the weights and biases of the layer
    void initialize_weights_and_biases(); // Function to initialize weights and biases

    

    ~LinearLayer(); // Destructor

private:

};

#endif