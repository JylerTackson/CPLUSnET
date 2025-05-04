// Tensor Implementation Header File
//  Contains the defintion of the TensorImpl class that implements the Tensor Object.

// Responsible for:
//  - Data storage and management
//  - Memory management
//  - Tensor operations (addition, multiplication, etc.)
//  - Tensor reshaping and slicing

#ifndef TENSORIMPL_HPP
#define TENSORIMPL_HPP

// Standard Libraries used within this file
#include <algorithm>
// https://en.cppreference.com/w/cpp/header/algorithm
#include <atomic>
// https://en.cppreference.com/w/cpp/atomic/atomic
#include <cstddef>
// https://en.cppreference.com/w/cpp/header/cstddef
#include <cstdint>
// https://en.cppreference.com/w/cpp/header/cstdint
#include <limits>
// https://en.cppreference.com/w/cpp/header/limits
#include <memory>
// https://en.cppreference.com/w/cpp/header/memory
#include <string>
// https://en.cppreference.com/w/cpp/header/string
#include <type_traits>
// https://en.cppreference.com/w/cpp/header/type_traits
#include <utility>
// https://en.cppreference.com/w/cpp/header/utility
#include <vector>
// https://en.cppreference.com/w/cpp/header/vector

namespace Tens
{
    class Tensor;
    class TensorBase;
}

// It's common to use structs for types such as 'data containers' or 'data structures'.
// Truthfully there is no big difference between a struct and a class in C++ other than having private members by default in a class.
// However, it is common to use structs for data containers so that you can freely access them without getters and setters.
struct TensorImpl
{
    // PyTorch uses 3 differnet constructors:

    // Construct a 1-dim 0-size tensor backed by the given storage.
    TensorImpl(Storage &&storage,
               DipatchKeySet,
               const caffe2::TypeMeta &dtype, );

    // Skip second

    // Construct a 1-dim 0-size tensor that doesn't own storage.
    TensorImpl(DispatchKeySet,
               const caffe2::TypeMeta &dtype,
               std::optional</*Device*/> device_opt);
}
#endif
