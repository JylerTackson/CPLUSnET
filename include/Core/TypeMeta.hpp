// TypeMeta will be responsible for inferring the data type of the scalar values within the tensor.

// https://github.com/pytorch/pytorch/blob/main/c10/util/typeid.h

#ifndef TYPEMETA_HPP
#define TYPEMETA_HPP

// includes from PyTorch typeid.h file
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>
