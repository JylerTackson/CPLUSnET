// TypeMeta will be responsible for reading the data type being parse to the tensor.
// This will help with assiging the appropriate size to the tensor.

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

struct TypeMetaData final
{
    // Using is a keyword in C++ that allows you to create an alias for a type.
    // Here we are creating an alias for a function pointer type.
    using New = void *();
    using PlacementNew = void(void *, size_t);
    using Copy = void(const void *, void *, size_t);
    using PlacementDelete = void(void *, size_t);
    using Delete = void(void *);

    // Default constructor
    // constexpr means that the constructor can be evaluated at compile time.
    // noexcept promises construction never throws.
    constexpr TypeMetaData() noexcept
        : itemsize(0),
          new_(nullptr),
          placementNew_(nullptr),
          copy_(nullptr),
          palcementDelete_(nullptr),
          delete_(nullptr),
          id_(TypeIdentifier::uninitialized()),
          name_("nullptr (Uninitialized)") {}

    // Full Constructor
    constexpr TypeMetaData(
        size_t itemsize,
        New *new_,
        PlacementNew *placementNew_,
        Copy *copy_,
        PlacementDelete *placementDelete_,
        Delete *delete_,
        TypeIdentifier id_,
        const char *name_) noexcept
        : itemsize(itemsize),
          new_(newFn),
          placementNew_(placementNew),
          copy_(copy),
          palcementDelete_(placementDelete),
          delete_(deleteFn),
          id_(id),
          name_(name) {}

    size_t itemsize_; // sizeof(T)

    New *new_; // malloc+construct

    PlacementNew *placementNew_; // placement new

    Copy *copy_; // copy construct

    PlacementDelete *placementDelete_; // explicit destructor

    Delete *delete_; // destroy+free

    TypeIdentifier id_; // small, unique tag for type erasure

    std::string_view name_; // human‑readable type name
};
