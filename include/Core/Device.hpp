// Device class that will be responsible for managing/reporting the device that the memory is living on

// Will only create support for two Devices for now:
//  CPU: Systems main processor (RAM)
//  GPU: NVIDIA CUDA

#ifndef DEVICE_HPP
#define DEVICE_HPP

#pragma once
#include <bitset>
// https://en.cppreference.com/w/cpp/header/bitset
#include <cstdint>
#include <string>
#include <stdexcept>

// This class will be used to manage the device that the tensor is on
class Device
{
public:
    Device();
    ~Device();

private:
    // This Struct will keep track of the device for later use
    struct DipatchKeySet
    {
    };
};

#endif // DEVICE_HPP