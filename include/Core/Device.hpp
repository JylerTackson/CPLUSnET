// Device class that will be responsible for managing/reporting the device that the memory is living on

// Will only create support for two Devices for now:
//  CPU: Systems main processor (RAM)
//  GPU: NVIDIA CUDA

#ifndef DEVICE_HPP
#define DEVICE_HPP

// This class will be used to manage the device that the tensor is on
class Device
{
public:
    Device();
    ~Device();

private:
};

// This class will be used to manage the dispatch keys for the device
class DispatchKeySet
{
public:
    DispatchKeySet();
    ~DispatchKeySet();

private:
};

#endif // DEVICE_HPP