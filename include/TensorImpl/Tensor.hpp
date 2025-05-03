// User Interface Framework for the Tensor Object.

#ifndef Tensor_hpp
#define Tensor_hpp
#include "TensorBase.hpp"
#include "TensorImpl.hpp"

class Tens::Tensor : public Tens::TensorBase
{
public:
    Tensor();
    ~Tensor();

private:
};

#endif // Tensor_hpp