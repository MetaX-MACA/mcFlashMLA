#pragma once


namespace flash {

inline __host__ mcDeviceProp_t mcGetCurrentDeviceProperties() {
    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);
    return dprops;
}

inline __host__ int mcGetCurrentDeviceArch() {
    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);
    return dprops.major * 100 + dprops.minor;
}

}