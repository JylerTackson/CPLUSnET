// Device class that will be responsible for managing/reporting the device that the memory is living on

// Will only create support for two Devices for now:
//  CPU: Systems main processor (RAM)
//  GPU: NVIDIA CUDA

#ifndef DEVICE_HPP
#define DEVICE_HPP

#pragma once
#include <bitset>         // Provides bitset functionality
#include <cstdint>        // Provides integer types
#include <string>         // Provides string functionality
#include <sstream>        // Provides string stream functionality
#include <stdexcept>      // Provides standard exceptions
#include <cuda_runtime.h> // Provides CUDA runtime API functions
#include <thread>         // Provides thread functionality (CPU)

// contain all device related classes and functions
namespace Device
{

    //

    class Device final
    {
    public:
        // We use a struct because we want to be able to freely assign the bits of the struct
        struct DispatchKeySet
        {
            // enum is a keyword in C++ that allows you to define a data type.
            // Here we are defining an enum class called Key.
            // The enum class is a scoped enumeration within the DispatchKeySet struct.
            enum class Key : uint8_t
            {

                // If undefined we should throw error
                Undefined = 0,
                CPU = 1,
                CUDA = 2
            };

            // DispatchKeySet Class Constructors-------------------------------------

            // Default constructor initializes bits_ to 0
            constexpr DispatchKeySet() : bits_(0) {}

            // We are using keyword explicit to prevent implicit conversions.
            // Using our enumerated data type Key to set the bits_ variable.
            // The explicit keyword prevents implicit conversions from Key to DispatchKeySet.
            constexpr explicit DispatchKeySet(Key K) : bits_(1ULL << static_cast<uint8_t>(K)) {}
            // Key K will correspond to the device type (CPU = 1, CUDA = 2)
            // The static_cast<uint8_t>(K) converts the Key enum value to its underlying type (uint8_t).
            // The 1ULL << static_cast<uint8_t>(K) shifts the bit 1 to the left by the value of K,
            // resulting in a bitmask where only the bit corresponding to K is set to 1.

            // helper functions:--------------------------------------

            // Checks if the bits_ is currently set to Key.
            bool has(Key K) const {}

            // Returns the raw bits of the DispatchKeySet.
            uint64_t raw_bits() const { return bits_; }

        private:
            // 64-bit unsigned integer to store the bitset
            uint64_t bits_;

        }; // end of DispatchKeySet Struct
    }
} // end of Device Namespace
#endif // DEVICE_HPP