/*
 cu.hpp (The C++ Wrapper for CUDA Driver API)
 
 Copyright (c) 2013 peta.okechan.net
 
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef cu_hpp
#define cu_hpp

#include <exception>
#include <string>
#include <sstream>
#include <vector>
#include <CUDA/CUDA.h>

namespace cu {
    
    /*
     Dimension
     for gridDim and blockDim
     */
    class Dim3
    {
    public:
        unsigned int x, y, z;
        
        Dim3(const unsigned int _x, const unsigned int _y, const unsigned int _z)
        : x(_x), y(_y), z(_z)
        {}
        
        Dim3(const unsigned int *_f)
        : x(_f[0]), y(_f[1]), z(_f[2])
        {}
        
        ~Dim3() {}
    };
    
    /*
     CUresult to string
     */
    std::string ResultString(CUresult result)
    {
        static const char *resultStrings[] = {
            [CUDA_SUCCESS] = "CUDA_SUCCESS",
            [CUDA_ERROR_INVALID_VALUE] = "CUDA_ERROR_INVALID_VALUE",
            [CUDA_ERROR_OUT_OF_MEMORY] = "CUDA_ERROR_OUT_OF_MEMORY",
            [CUDA_ERROR_NOT_INITIALIZED] = "CUDA_ERROR_NOT_INITIALIZED",
            [CUDA_ERROR_DEINITIALIZED] = "CUDA_ERROR_DEINITIALIZED",
            [CUDA_ERROR_PROFILER_DISABLED] = "CUDA_ERROR_PROFILER_DISABLED",
            [CUDA_ERROR_PROFILER_NOT_INITIALIZED] = "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
            [CUDA_ERROR_PROFILER_ALREADY_STARTED] = "CUDA_ERROR_PROFILER_ALREADY_STARTED",
            [CUDA_ERROR_PROFILER_ALREADY_STOPPED] = "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
            [CUDA_ERROR_NO_DEVICE] = "CUDA_ERROR_NO_DEVICE",
            [CUDA_ERROR_INVALID_DEVICE] = "CUDA_ERROR_INVALID_DEVICE",
            [CUDA_ERROR_INVALID_IMAGE] = "CUDA_ERROR_INVALID_IMAGE",
            [CUDA_ERROR_INVALID_CONTEXT] = "CUDA_ERROR_INVALID_CONTEXT",
            [CUDA_ERROR_CONTEXT_ALREADY_CURRENT] = "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            [CUDA_ERROR_MAP_FAILED] = "CUDA_ERROR_MAP_FAILED",
            [CUDA_ERROR_UNMAP_FAILED] = "CUDA_ERROR_UNMAP_FAILED",
            [CUDA_ERROR_ARRAY_IS_MAPPED] = "CUDA_ERROR_ARRAY_IS_MAPPED",
            [CUDA_ERROR_ALREADY_MAPPED] = "CUDA_ERROR_ALREADY_MAPPED",
            [CUDA_ERROR_NO_BINARY_FOR_GPU] = "CUDA_ERROR_NO_BINARY_FOR_GPU",
            [CUDA_ERROR_ALREADY_ACQUIRED] = "CUDA_ERROR_ALREADY_ACQUIRED",
            [CUDA_ERROR_NOT_MAPPED] = "CUDA_ERROR_NOT_MAPPED",
            [CUDA_ERROR_NOT_MAPPED_AS_ARRAY] = "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
            [CUDA_ERROR_NOT_MAPPED_AS_POINTER] = "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
            [CUDA_ERROR_ECC_UNCORRECTABLE] = "CUDA_ERROR_ECC_UNCORRECTABLE",
            [CUDA_ERROR_UNSUPPORTED_LIMIT] = "CUDA_ERROR_UNSUPPORTED_LIMIT",
            [CUDA_ERROR_CONTEXT_ALREADY_IN_USE] = "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
            [CUDA_ERROR_PEER_ACCESS_UNSUPPORTED] = "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
            [CUDA_ERROR_INVALID_SOURCE] = "CUDA_ERROR_INVALID_SOURCE",
            [CUDA_ERROR_FILE_NOT_FOUND] = "CUDA_ERROR_FILE_NOT_FOUND",
            [CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND] = "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
            [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED] = "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
            [CUDA_ERROR_OPERATING_SYSTEM] = "CUDA_ERROR_OPERATING_SYSTEM",
            [CUDA_ERROR_INVALID_HANDLE] = "CUDA_ERROR_INVALID_HANDLE",
            [CUDA_ERROR_NOT_FOUND] = "CUDA_ERROR_NOT_FOUND",
            [CUDA_ERROR_NOT_READY] = "CUDA_ERROR_NOT_READY",
            [CUDA_ERROR_LAUNCH_FAILED] = "CUDA_ERROR_LAUNCH_FAILED",
            [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES] = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            [CUDA_ERROR_LAUNCH_TIMEOUT] = "CUDA_ERROR_LAUNCH_TIMEOUT",
            [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING] = "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
            [CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED] = "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
            [CUDA_ERROR_PEER_ACCESS_NOT_ENABLED] = "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
            [CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE] = "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
            [CUDA_ERROR_CONTEXT_IS_DESTROYED] = "CUDA_ERROR_CONTEXT_IS_DESTROYED",
            [CUDA_ERROR_ASSERT] = "CUDA_ERROR_ASSERT",
            [CUDA_ERROR_TOO_MANY_PEERS] = "CUDA_ERROR_TOO_MANY_PEERS",
            [CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED] = "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
            [CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED] = "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
            [CUDA_ERROR_NOT_PERMITTED] = "CUDA_ERROR_NOT_PERMITTED",
            [CUDA_ERROR_NOT_SUPPORTED] = "CUDA_ERROR_NOT_SUPPORTED",
            [CUDA_ERROR_UNKNOWN] = "CUDA_ERROR_UNKNOWN",
        };
        
        if (result < CUDA_SUCCESS || result > CUDA_ERROR_UNKNOWN) {
            std::ostringstream s;
            s << "Out of CUResult range(" << result << ")";
            return s.str();
        }
        
        if (resultStrings[result]) {
            return resultStrings[result];
        } else {
            std::ostringstream s;
            s << "No result string(" << result << ")";
            return s.str();
        }
    }
    
    /*
     Exception class
     */
    class Error : public std::exception
    {
    private:
        CUresult _err;
        std::string _errString;
    public:
        Error(CUresult err, const std::string &errString = nullptr) throw()
        : _err(err), _errString(errString)
        {}
        
        ~Error() throw() {}
        
        const char* what() const throw()
        {
            return _errString.c_str();
        }
        
        std::string string() const throw()
        {
            return _errString;
        }
        
        CUresult code(void) const
        {
            return _err;
        }
        
        static void Check(CUresult result)
        {
            if (result != CUDA_SUCCESS) {
                std::string str = ResultString(result);
                throw Error(result, str);
            }
        }
    };
    
    /*
     cuInit wrapper
     */
    void Init(unsigned int flags = 0)
    {
        Error::Check(cuInit(flags));
    }
    
    /*
     cuDriverGetVersion wrapper
     */
    int GetDriverVersion()
    {
        int ret = 0;
        Error::Check(cuDriverGetVersion(&ret));
        return ret;
    }
    
    /*
     cuCtxGetCacheConfig wrapper
     */
    CUfunc_cache GetCtxCacheConfig()
    {
        CUfunc_cache config;
        Error::Check(cuCtxGetCacheConfig(&config));
        return config;
    }
    
    /*
     cuCtxSynchronize wrapper
     */
    void CtxSynchronize()
    {
        Error::Check(cuCtxSynchronize());
    }
    
    /*
     CUdevice wrapper
     */
    class Device
    {
    private:
        CUdevice device;
    public:
        Device(CUdevice _device)
        : device(_device)
        {}
        
        ~Device() {}
        
        CUdevice operator()(void) const
        {
            return device;
        }
        
        std::string getName()
        {
            char name[100];
            int len = 100;
            Error::Check(cuDeviceGetName(name, len, device));
            return std::string(name);
        }
        
        size_t getTotalMemBytes()
        {
            size_t bytes = 0;
            Error::Check(cuDeviceTotalMem(&bytes, device));
            return bytes;
        }
        
        template<CUdevice_attribute attr>
        int getAttribute()
        {
            int ret = 0;
            Error::Check(cuDeviceGetAttribute(&ret, attr, device));
            return ret;
        }
        
        static std::vector<Device> GetDevices()
        {
            int count = 0;
            Error::Check(cuDeviceGetCount(&count));
            
            std::vector<Device> devices;
            devices.reserve(count);
            for (int i = 0; i < count; i++) {
                CUdevice device;
                Error::Check(cuDeviceGet(&device, i));
                devices.push_back(Device(device));
            }
            
            return devices;
        }
    };
    
    /*
     CUcontext wrapper
     */
    class Context
    {
    private:
        CUcontext ctx;
    public:
        Context(Device device, unsigned int flags = CU_CTX_SCHED_AUTO)
        {
            Error::Check(cuCtxCreate(&ctx, flags, device()));
        }
        
        ~Context()
        {
            Error::Check(cuCtxDestroy(ctx));
        }
        
        CUcontext operator()(void) const
        {
            return ctx;
        }
        
        unsigned int getApiVersion()
        {
            unsigned int version;
            Error::Check(cuCtxGetApiVersion(ctx, &version));
            return version;
        }
        
        void getMemInfo(size_t &memFree, size_t &memTotal) const
        {
            Error::Check(cuMemGetInfo(&memFree, &memTotal));
        }
    };
    
    /*
     CUmodule wrapper
     */
    class Module
    {
    private:
        CUmodule mod;
    public:
        Module(const std::string modulePath)
        {
            Error::Check(cuModuleLoad(&mod, modulePath.c_str()));
        }
        
        ~Module()
        {
            Error::Check(cuModuleUnload(mod));
        }
        
        CUmodule operator()(void) const
        {
            return mod;
        }
    };
    
    /*
     CUdeviceptr wrapper
     */
    class Memory
    {
    private:
        CUdeviceptr ptr;
        size_t byteCount;
    public:
        Memory(const size_t _byteCount)
        : byteCount(_byteCount)
        {
            Error::Check(cuMemAlloc(&ptr, _byteCount));
        }
        
        ~Memory()
        {
            Error::Check(cuMemFree(ptr));
        }
        
        CUdeviceptr operator()(void) const
        {
            return ptr;
        }
        
        CUdeviceptr* operator()(void)
        {
            return &ptr;
        }
        
        size_t size(void) const
        {
            return byteCount;
        }
        
        void copyFrom(void* srcHost)
        {
            Error::Check(cuMemcpyHtoD(ptr, srcHost, byteCount));
        }
        
        void copyFrom(void* srcHost, size_t _byteCount)
        {
            Error::Check(cuMemcpyHtoD(ptr, srcHost, _byteCount));
        }
        
        template<typename T>
        void copyFrom(const std::vector<T> &_vector)
        {
            Error::Check(cuMemcpyHtoD(ptr, _vector.data(), sizeof(T) * _vector.size()));
        }
        
        void copyTo(void* dstHost)
        {
            Error::Check(cuMemcpyDtoH(dstHost, ptr, byteCount));
        }
        
        void copyTo(void* dstHost, size_t _byteCount)
        {
            Error::Check(cuMemcpyDtoH(dstHost, ptr, _byteCount));
        }
        
        template<typename T>
        void copyTo(std::vector<T> &_vector) const
        {
            Error::Check(cuMemcpyDtoH(_vector.data(), ptr, sizeof(T) * _vector.size()));
        }
    };
    
    /*
     CUtexref wrapper
     */
    class TexRef
    {
    private:
        CUtexref ref;
    public:
        TexRef(const Module &mod, const std::string name)
        {
            Error::Check(cuModuleGetTexRef(&ref, mod(), name.c_str()));
        }
        
        ~TexRef() {}
        
        void setMemory(const Memory &mem)
        {
            Error::Check(cuTexRefSetAddress(nullptr, ref, mem(), mem.size()));
        }
        
        void setAddressMode(const int dim, const CUaddress_mode mode)
        {
            Error::Check(cuTexRefSetAddressMode(ref, dim, mode));
        }
        
        void setFilterMode(const CUfilter_mode mode)
        {
            Error::Check(cuTexRefSetFilterMode(ref, mode));
        }
        
        void setFlags(unsigned int flags)
        {
            Error::Check(cuTexRefSetFlags(ref, flags));
        }
        
        void setFormat(const CUarray_format format, const int numPackedComponents)
        {
            Error::Check(cuTexRefSetFormat(ref, format, numPackedComponents));
        }
    };
    
    /*
     CUfunction wrapper
     */
    class Function
    {
    private:
        CUfunction func;
        std::vector<void*> args;
    public:
        Function(const Module &mod, const std::string name)
        {
            Error::Check(cuModuleGetFunction(&func, mod(), name.c_str()));
        }
        
        ~Function() {}
        
        CUfunction operator()(void) const
        {
            return func;
        }
        
        template<typename T>
        Function& setArg(T &value)
        {
            args.push_back(&value);
            return *this;
        }
        
        Function& setArg(Memory &value)
        {
            args.push_back(value());
            return *this;
        }
        
        void launchKernel(const Dim3 &gridDim, const Dim3 &blockDim, unsigned int sharedMemBytes = 0, CUstream hStream = nullptr, void** extra = nullptr)
        {
            Error::Check(cuLaunchKernel(func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMemBytes, hStream, args.data(), extra));
        }
    };
    
    /*
     CUevent wrapper
     */
    class Event
    {
    private:
        CUevent e;
    public:
        Event(unsigned int flags = CU_EVENT_DEFAULT)
        {
            Error::Check(cuEventCreate(&e, flags));
        }
        
        ~Event()
        {
            Error::Check(cuEventDestroy(e));
        }
        
        CUevent operator()(void) const
        {
            return e;
        }
        
        void record(const CUstream &hStream = 0)
        {
            Error::Check(cuEventRecord(e, hStream));
        }
        
        void query()
        {
            Error::Check(cuEventQuery(e));
        }
        
        void synchronize()
        {
            Error::Check(cuEventSynchronize(e));
        }
    };
    
    /*
     begin - end CUevent wrapper
     */
    class Timer
    {
    private:
        Event begin, end;
    public:
        Timer(unsigned int flags = CU_EVENT_DEFAULT)
        : begin(flags), end(flags)
        {}
        
        ~Timer() {}
        
        void start(const CUstream &hStream = 0)
        {
            CtxSynchronize();
            begin.record(hStream);
        }
        
        void stop(const CUstream &hStream = 0)
        {
            CtxSynchronize();
            end.record(hStream);
            end.synchronize();
        }
        
        float elapsedMilliSec()
        {
            float msec;
            Error::Check(cuEventElapsedTime(&msec, begin(), end()));
            return msec;
        }
    };
}

#endif
