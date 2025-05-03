//Pytorch is built with C++14 standard version.
//PyTorch uses Python as its primary interface, but it is built on top of C++ for performance reasons.
//The core libraries and many of the underlying operations are implemented in C++ for efficiency.
//The Python interface provides a more user-friendly way to interact with the library, while the C++ backend handles the heavy lifting.

//PyTorch Conv Module:
// https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py

//PyTorch Conv.cpp:
//https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/conv.cpp

#ifndef CONV_H
#define CONV_H

class Convolutiona_1d {
    public:
        Convolutiona_1d();
        ~Convolutiona_1d();

    private:
        // Convolutional Layer Attributes:
        //----------------------------------------------------------------------------
        int input_channels;   // Number of input channels (e.g., RGB channels for images)
        int output_channels;  // Number of output channels (number of filters)
        int kernel_size;      // Size of the convolutional kernel/filter
        int stride;           // Stride of the convolution operation
        int padding;          // Padding added to the input
};

class Convolutional_2d {
    public:
        Convolutional_2d();
        ~Convolutional_2d();
    
    private:
    // Convolutional Layer Attributes:
        //----------------------------------------------------------------------------
        int input_channels;   // Number of input channels (e.g., RGB channels for images)
        int output_channels;  // Number of output channels (number of filters)
        int kernel_size;      // Size of the convolutional kernel/filter
        int stride;           // Stride of the convolution operation
        int padding;          // Padding added to the input
};

#endif // CONV_H