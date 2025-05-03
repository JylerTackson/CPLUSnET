Pytorch is built with C++14 standard version.
PyTorch uses Python as its primary interface, but it is built on top of C++ for performance reasons.
The core libraries and many of the underlying operations are implemented in C++ for efficiency.
The Python interface provides a more user-friendly way to interact with the library, while the C++ backend handles the heavy lifting.

pytorch/
│
├── aten/ **# Core tensor library (C++ backend)**

The code that we need to be reviewing to create our tensor class and our gradient engine is within the aten folder within the PyTorch github:

- https://github.com/pytorch/pytorch/tree/main/aten/src

### 1. AutoGrad Gradient Engine:

##### PyTorch Code:

- Core execution engine:
  - https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.h
  - https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.cpp
- Tensor Wrapper to track gradients:
  - https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/variable.h
  - https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/variable.cpp

### 2. Tensor Implementation:

##### W3Schools:

- https://www.w3schools.com/ai/ai_tensors.asp

##### PyTorch Code:

The TensorImpl class is the actual defintion of the Tensor Object that is used within the PyTorch Engines (Raw data + metadata):

- https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h
- https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.cpp

At the beginning of TensorImpl they create a NameSpace 'at' and declare two classes:

- **TensorBase** - Line 82

  - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/TensorBase.h
    This class is an abstraction layer that contains a smart pointer to TensorImpl; Provides basic read/write APIs as well as operator overlaods.

- Tensor - Line 92
  - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/templates/TensorBody.h
    This class is the public facing object and **inherits** from the TensorBase class; Gives the object a user interface to control the object.

These classes belong to the name space but are defined in seperate header files.
The decleration in the namespace at the top of TensorImpl.h does not DEFINE the function it just DECLARES them to the compiler meanwhile the actual definition of the Tensor Classes happens within the corresponding header files.

- Supplemental:
  - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Tensor.h
  - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Tensor.cpp

### 3. Loss Functions:

##### PyTorch Code:

- Registered through native_functions.yaml. (No Header File)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp

### 4. Activation Functions:

##### PyTorch Code:

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.h
- CPU:
  - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp
- CUDA:
  - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationGeluKernel.cu

### 5. Padding Operators:

##### PyTorch Code:

- Registered through native_functions.yaml. (No Header File)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ReflectionPad.cpp
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ReplicationPadding.cpp

### 6. Pooling Operators:

##### PyTorch Code:

- These kernels are registered through native_functions.yaml, so only .cpp files exist.
- https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Pooling.cpp
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/AveragePool2d.cpp

### 7. Convolution:

##### PyTorch Code:

- Registered through native_functions.yaml. (No Header File)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp

### 8. Linear:

##### PyTorch Code:

- https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/nn/modules/linear.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Linear.cpp
- https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/nn/modules/linear.cpp

### 9. Sequential:

##### PyTorch Code:

- https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/nn/modules/container/sequential.h
- All Logic is inline within the header file. Allows for compile-time flexibility.

**What is a .yaml file?**

- YAML stands for "YAML Ain't Markup Language". It is a human-readable data serialization format, similar to JSON but cleaner and more structured for configuration files. YAML uses indentation to represent structure. It’s commonly used for configuration files (e.g., Docker, GitHub Actions) or data input to programs.
- PyTorch uses .yaml files to declare and register operators, especially in the C++ backend, so the system can auto-generate header files, set up dispatcher routing (CPU/CUDA), and bind Python to C++ function calls.
