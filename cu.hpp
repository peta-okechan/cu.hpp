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

#include <iostream>
#include <exception>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <cassert>
#if defined(__APPLE__) || defined(__MACOSX)
#include <CUDA/CUDA.h>
#else
#include <cuda.h>
#endif

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
     Format size
     */
    size_t FormatSize(const CUarray_format fmt)
    {
        size_t csize = 0;
        switch (fmt) {
            case CU_AD_FORMAT_UNSIGNED_INT8:
                csize = sizeof(uint8_t);
                break;
                
            case CU_AD_FORMAT_UNSIGNED_INT16:
                csize = sizeof(uint16_t);
                break;
                
            case CU_AD_FORMAT_UNSIGNED_INT32:
                csize = sizeof(uint32_t);
                break;
                
            case CU_AD_FORMAT_SIGNED_INT8:
                csize = sizeof(int8_t);
                break;
                
            case CU_AD_FORMAT_SIGNED_INT16:
                csize = sizeof(int16_t);
                break;
                
            case CU_AD_FORMAT_SIGNED_INT32:
                csize = sizeof(int32_t);
                break;
                
            case CU_AD_FORMAT_HALF:
                csize = sizeof(float) / 2;
                break;
                
            case CU_AD_FORMAT_FLOAT:
                csize = sizeof(float);
                break;
                
            default:
                throw Error(CUDA_ERROR_UNKNOWN, "Invalid format.");
                break;
        }
        return csize;
    }
    
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
     cuCtxGetCurrent wrapper
     */
    CUcontext GetCtxCurrent()
    {
        CUcontext ctx;
        Error::Check(cuCtxGetCurrent(&ctx));
        return ctx;
    }
    
    /*
     cuCtxGetDevice wrapper
     */
    CUdevice GetCtxDevice()
    {
        CUdevice dev;
        Error::Check(cuCtxGetDevice(&dev));
        return dev;
    }
    
    /*
     cuCtxGetLimit wrapper
     */
    template<CUlimit limit>
    size_t GetCtxLimit()
    {
        size_t pvalue;
        Error::Check(cuCtxGetLimit(&pvalue, limit));
        return pvalue;
    }
    
    /*
     cuCtxGetSharedMemConfig wrapper
     */
    CUsharedconfig GetCtxSharedMemConfig()
    {
        CUsharedconfig pconfig;
        Error::Check(cuCtxGetSharedMemConfig(&pconfig));
        return pconfig;
    }
    
    /*
     cuCtxSetCacheConfig wrapper
     */
    void SetCtxCacheConfig(CUfunc_cache config)
    {
        Error::Check(cuCtxSetCacheConfig(config));
    }
    
    /*
     cuCtxSetCurrent wrapper
     */
    void SetCtxCurrent(CUcontext ctx)
    {
        Error::Check(cuCtxSetCurrent(ctx));
    }
    
    /*
     cuCtxSetLimit wrapper
     */
    template<CUlimit limit>
    void SetCtxLimit(size_t value)
    {
        Error::Check(cuCtxSetLimit(limit, value));
    }
    
    /*
     cuCtxSetSharedMemConfig wrapper
     */
    void SetCtxSharedMemConfig(CUsharedconfig config)
    {
        Error::Check(cuCtxSetSharedMemConfig(config));
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
        friend class Context;
        
    private:
        std::shared_ptr<CUdevice> m_device;
        
        Device()
        : m_device()
        {
            throw "Do not use default ctor.";
        }
        
        Device(CUdevice a_device)
        : m_device()
        {
            m_device = std::make_shared<CUdevice>(a_device);
        }
        
    public:
        std::string getName() const
        {
            char name[100];
            int len = 100;
            Error::Check(cuDeviceGetName(name, len, *m_device));
            return std::move(std::string(name));
        }
        
        size_t getTotalMemBytes() const
        {
            size_t bytes = 0;
            Error::Check(cuDeviceTotalMem(&bytes, *m_device));
            return bytes;
        }
        
        template<CUdevice_attribute attr>
        int getAttribute() const
        {
            int ret = 0;
            Error::Check(cuDeviceGetAttribute(&ret, attr, *m_device));
            return ret;
        }
        
        static std::vector<Device> all()
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
            
            return std::move(devices);
        }
    };
    
    /*
     CUcontext wrapper
     */
    class Context
    {
    private:
        Device m_device;
        std::shared_ptr<CUctx_st> m_context;
        
        Context()
        : m_context()
        {
            throw "Do not use default ctor.";
        }
        
        Context(CUcontext a_context, Device a_device)
        : m_context(), m_device(a_device)
        {
            m_context = std::shared_ptr<CUctx_st>(a_context, [](CUctx_st *ctx){
                cuCtxDestroy(ctx);
#ifdef DEBUG
                std::cout << "cuCtxDestroy called." << std::endl;
#endif
            });
        }
        
    public:
        Context(Device a_device, unsigned int flags = CU_CTX_SCHED_AUTO)
        : m_device(a_device)
        {
            CUcontext ctx;
            Error::Check(cuCtxCreate(&ctx, flags, *(m_device.m_device)));
#ifdef DEBUG
            std::cout << "cuCtxCreate called." << std::endl;
#endif
            m_context = std::shared_ptr<CUctx_st>(ctx, [](CUctx_st *ctx){
                cuCtxDestroy(ctx);
#ifdef DEBUG
                std::cout << "cuCtxDestroy called." << std::endl;
#endif
            });
        }
        
        bool operator==(const Context &rhs) const
        {
            return (m_context == rhs.m_context);
        }
        
        void setCurrent() const
        {
            Error::Check(cuCtxSetCurrent(m_context.get()));
        }
        
        bool isCurrent() const
        {
            CUcontext ctx;
            Error::Check(cuCtxGetCurrent(&ctx));
            return m_context.get() == ctx;
        }
        
        CUfunc_cache getCacheConfig() const
        {
            assert(isCurrent());
            
            CUfunc_cache config;
            Error::Check(cuCtxGetCacheConfig(&config));
            return config;
        }
        
        Device getDevice() const
        {
            return m_device;
        }
        
        template<CUlimit limit>
        size_t getLimit() const
        {
            assert(isCurrent());
            
            size_t pvalue;
            Error::Check(cuCtxGetLimit(&pvalue, limit));
            return pvalue;
        }
        
        CUsharedconfig getSharedMemConfig() const
        {
            assert(isCurrent());
            
            CUsharedconfig pconfig;
            Error::Check(cuCtxGetSharedMemConfig(&pconfig));
            return pconfig;
        }
        
        void setCacheConfig(CUfunc_cache config)
        {
            assert(isCurrent());
            
            Error::Check(cuCtxSetCacheConfig(config));
        }
        
        template<CUlimit limit>
        void setLimit(size_t value)
        {
            assert(isCurrent());
            
            Error::Check(cuCtxSetLimit(limit, value));
        }
        
        void getSharedMemConfig(CUsharedconfig config) const
        {
            assert(isCurrent());
            
            Error::Check(cuCtxSetSharedMemConfig(config));
        }
        
        void synchronize() const
        {
            assert(isCurrent());
            
            Error::Check(cuCtxSynchronize());
        }
        
        unsigned int getApiVersion() const
        {
            unsigned int version;
            Error::Check(cuCtxGetApiVersion(m_context.get(), &version));
            return version;
        }
        
        void getMemInfo(size_t &memFree, size_t &memTotal) const
        {
            assert(isCurrent());
            
            Error::Check(cuMemGetInfo(&memFree, &memTotal));
        }
    };
    
    /*
     CUmodule wrapper
     */
    class Module
    {
        friend class Memory;
        friend class TexRef;
        friend class Function;
        
    private:
        std::shared_ptr<CUmod_st> m_module;
        
        Module()
        : m_module()
        {
            throw "Do not use default ctor.";
        }
        
        Module(CUmodule a_module)
        : m_module()
        {
            m_module = std::shared_ptr<CUmod_st>(a_module, [](CUmod_st *mod){
                cuModuleUnload(mod);
#ifdef DEBUG
                std::cout << "cuModuleUnload called." << std::endl;
#endif
            });
        }
        
    public:
        static Module loadFromFile(const std::string &modulePath)
        {
            CUmodule mod;
            Error::Check(cuModuleLoad(&mod, modulePath.c_str()));
#ifdef DEBUG
            std::cout << "cuModuleLoad called." << std::endl;
#endif
            return Module(mod);
        }
        
        static Module loadFromImage(const void *image)
        {
            CUmodule mod;
            Error::Check(cuModuleLoadData(&mod, image));
#ifdef DEBUG
            std::cout << "cuModuleLoadData called." << std::endl;
#endif
            return Module(mod);
        }
        
        static Module loadFromFatBinary(const void *fatCubin)
        {
            CUmodule mod;
            Error::Check(cuModuleLoadFatBinary(&mod, fatCubin));
#ifdef DEBUG
            std::cout << "cuModuleLoadFatBinary called." << std::endl;
#endif
            return Module(mod);
        }
    };
    
    /*
     Memcpy forward decls.
     */
    class Memory;
    class Array;
    void Memcpy(Array &dst, const Array &src, const size_t byteCount = 0, const size_t dstOffset = 0, const size_t srcOffset = 0);
    void Memcpy(Memory &dst, const Array &src, const size_t byteCount = 0, const size_t srcOffset = 0);
    void Memcpy(void *dst, const Array &src, const size_t byteCount = 0, const size_t srcOffset = 0);
    template<typename T>
    void Memcpy(std::vector<T> &dst, const Array &src, const size_t byteCount = 0, const size_t srcOffset = 0);
    void Memcpy(Array &dst, const Memory &src, const size_t byteCount = 0, const size_t dstOffset = 0);
    void Memcpy(Memory &dst, const Memory &src, const size_t byteCount = 0);
    void Memcpy(void *dst, const Memory &src, const size_t byteCount = 0);
    template<typename T>
    void Memcpy(std::vector<T> &dst, const Memory &src, const size_t byteCount = 0);
    void Memcpy(Array &dst, const void *src, const size_t byteCount = 0, const size_t dstOffset = 0);
    template<typename T>
    void Memcpy(Array &dst, const std::vector<T> &src, const size_t byteCount = 0, const size_t dstOffset = 0);
    void Memcpy(Memory &dst, const void *src, const size_t byteCount = 0);
    template<typename T>
    void Memcpy(Memory &dst, const std::vector<T> &src, const size_t byteCount = 0);
    
    /*
     CUdeviceptr wrapper
     */
    class Memory
    {
        friend class TexRef;
        friend class Function;
        friend void Memcpy(Memory &dst, const Array &src, const size_t byteCount, const size_t srcOffset);
        friend void Memcpy(Array &dst, const Memory &src, const size_t byteCount, const size_t dstOffset);
        friend void Memcpy(Memory &dst, const Memory &src, const size_t byteCount);
        friend void Memcpy(void *dst, const Memory &src, const size_t byteCount);
        template<typename T> friend void Memcpy(std::vector<T> &dst, const Memory &src, const size_t byteCount);
        friend void Memcpy(Memory &dst, const void *src, const size_t byteCount);
        template<typename T> friend void Memcpy(Memory &dst, const std::vector<T> &src, const size_t byteCount);
        
    private:
        std::shared_ptr<CUdeviceptr> m_devptr;
        size_t m_byteCount;
        
        Memory()
        : m_devptr()
        {
            throw "Do not use default ctor.";
        }
        
        Memory(CUdeviceptr a_devptr, size_t const a_byteCount)
        : m_devptr(), m_byteCount(a_byteCount)
        {
            m_devptr = std::shared_ptr<CUdeviceptr>(new CUdeviceptr(a_devptr), [](CUdeviceptr *devptr){
                cuMemFree(*devptr);
#ifdef DEBUG
                std::cout << "cuMemFree called." << std::endl;
#endif
            });
        }
        
    public:
        Memory(const size_t a_byteCount)
        : m_byteCount(a_byteCount)
        {
            CUdeviceptr devptr;
            Error::Check(cuMemAlloc(&devptr, m_byteCount));
#ifdef DEBUG
            std::cout << "cuMemAlloc called." << std::endl;
#endif
            m_devptr = std::shared_ptr<CUdeviceptr>(new CUdeviceptr(devptr), [](CUdeviceptr *devptr){
                cuMemFree(*devptr);
#ifdef DEBUG
                std::cout << "cuMemFree called." << std::endl;
#endif
            });
        }
        
        Memory(const Module &mod, const std::string name)
        {
            CUdeviceptr devptr;
            Error::Check(cuModuleGetGlobal(&devptr, &m_byteCount, mod.m_module.get(), name.c_str()));
            m_devptr = std::make_shared<CUdeviceptr>(devptr);
            /*
             cuMemAlloc() と cuMemAllocPitch() 以外で取得した CUdeviceptr は
             cuMemFree() で解放する必要はない（解放できない）ので
             カスタムデリータは指定しない
             */
        }
        
        size_t getTotalBytes(void) const
        {
            return m_byteCount;
        }
    };
    
    /*
     Array Descriptor
     */
    class Descriptor
    {
        friend class Array;
        
    private:
        bool is3D;
        size_t width, height, depth;
        CUarray_format format;
        unsigned int numChannels, flags;
        
        Descriptor()
        {
            
        }
        
    public:
        Descriptor(const bool _is3D, const size_t _width, const size_t _height, const size_t _depth, const CUarray_format _format, const unsigned int _numChannels, const unsigned int _flags)
        : is3D(_is3D), width(_width), height(_height), depth(_depth), format(_format), numChannels(_numChannels), flags(_flags)
        {}
        
        Descriptor(const CUDA_ARRAY_DESCRIPTOR &desc)
        : is3D(false), width(desc.Width), height(desc.Height), depth(1), format(desc.Format), numChannels(desc.NumChannels), flags(0)
        {}
        
        Descriptor(const CUDA_ARRAY3D_DESCRIPTOR &desc)
        : is3D(false), width(desc.Width), height(desc.Height), depth(desc.Depth), format(desc.Format), numChannels(desc.NumChannels), flags(desc.Flags)
        {}
        
        size_t getTotalBytes() const
        {
            return FormatSize(format) * numChannels * width * height * depth;
        }
        
        void getDescriptor(CUDA_ARRAY_DESCRIPTOR &desc) const
        {
            if (is3D) throw Error(CUDA_ERROR_UNKNOWN, "This is not 1D or 2D descriptor.");
            desc.Width = width;
            desc.Height = height;
            desc.Format = format;
            desc.NumChannels = numChannels;
        }
        
        void getDescriptor(CUDA_ARRAY3D_DESCRIPTOR &desc) const
        {
            if (!is3D) throw Error(CUDA_ERROR_UNKNOWN, "This is not 3D descriptor.");
            desc.Width = width;
            desc.Height = height;
            desc.Depth = depth;
            desc.Format = format;
            desc.NumChannels = numChannels;
            desc.Flags = flags;
        }
        
        void update(const CUarray handle)
        {
            if (is3D) {
                CUDA_ARRAY3D_DESCRIPTOR _desc;
                Error::Check(cuArray3DGetDescriptor(&_desc, handle));
                width = _desc.Width;
                height = _desc.Height;
                format = _desc.Format;
                numChannels = _desc.NumChannels;
                depth = _desc.Depth;
                flags = _desc.Flags;
            } else {
                CUDA_ARRAY_DESCRIPTOR _desc;
                Error::Check(cuArrayGetDescriptor(&_desc, handle));
                width = _desc.Width;
                height = _desc.Height;
                format = _desc.Format;
                numChannels = _desc.NumChannels;
            }
        }
    };
    
    /*
     CUarray wrapper
     */
    class Array
    {
        friend class TexRef;
        friend void Memcpy(Array &dst, const Array &src, const size_t byteCount, const size_t dstOffset, const size_t srcOffset);
        friend void Memcpy(Memory &dst, const Array &src, const size_t byteCount, const size_t srcOffset);
        friend void Memcpy(void *dst, const Array &src, const size_t byteCount, const size_t srcOffset);
        template<typename T> friend void Memcpy(std::vector<T> &dst, const Array &src, const size_t byteCount, const size_t srcOffset);
        friend void Memcpy(Array &dst, const Memory &src, const size_t byteCount, const size_t dstOffset);
        friend void Memcpy(Array &dst, const void *src, const size_t byteCount, const size_t dstOffset);
        template<typename T> friend void Memcpy(Array &dst, const std::vector<T> &src, const size_t byteCount, const size_t dstOffset);
        
    private:
        std::shared_ptr<CUarray_st> m_array;
        Descriptor m_descriptor;
        
        Array()
        : m_array()
        {
            throw "Do not use default ctor.";
        }
        
        Array(CUarray a_array, Descriptor const &a_descriptor)
        : m_array(), m_descriptor(a_descriptor)
        {
            m_array = std::shared_ptr<CUarray_st>(a_array, [](CUarray_st *array){
                cuArrayDestroy(array);
#ifdef DEBUG
                std::cout << "cuArrayDestroy called." << std::endl;
#endif
            });
        }
        
    public:
        Array(const Descriptor &a_descriptor)
        : m_descriptor(a_descriptor)
        {
            CUarray array;
            if (m_descriptor.is3D) {
                CUDA_ARRAY3D_DESCRIPTOR desc;
                m_descriptor.getDescriptor(desc);
                Error::Check(cuArray3DCreate(&array, &desc));
#ifdef DEBUG
                std::cout << "cuArray3DCreate called." << std::endl;
#endif
            } else {
                CUDA_ARRAY_DESCRIPTOR desc;
                m_descriptor.getDescriptor(desc);
                Error::Check(cuArrayCreate(&array, &desc));
#ifdef DEBUG
                std::cout << "cuArrayCreate called." << std::endl;
#endif
            }
            m_array = std::shared_ptr<CUarray_st>(array, [](CUarray_st *array){
                cuArrayDestroy(array);
#ifdef DEBUG
                std::cout << "cuArrayDestroy called." << std::endl;
#endif
            });
        }
        
        Descriptor getDescriptor(const bool useCache = true)
        {
            if (!useCache) {
                m_descriptor.update(m_array.get());
            }
            return m_descriptor;
        }
        
        size_t getTotalBytes() const
        {
            return m_descriptor.getTotalBytes();
        }
        
        static Array Create1D(const CUarray_format Format, const unsigned int NumChannels, const unsigned int width)
        {
            CUDA_ARRAY_DESCRIPTOR desc;
            desc.Width = width;
            desc.Height = 1;
            desc.Format = Format;
            desc.NumChannels = NumChannels;
            return Array(Descriptor(desc));
        }
        
        static Array Create2D(const CUarray_format Format, const unsigned int NumChannels, const unsigned int width, const unsigned int height)
        {
            CUDA_ARRAY_DESCRIPTOR desc;
            desc.Width = width;
            desc.Height = height;
            desc.Format = Format;
            desc.NumChannels = NumChannels;
            return Array(Descriptor(desc));
        }
        
        static Array Create3D(const CUarray_format Format, const unsigned int NumChannels, const unsigned int width, const unsigned int height, const unsigned int depth, unsigned int flags)
        {
            CUDA_ARRAY3D_DESCRIPTOR desc;
            desc.Width = width;
            desc.Height = height;
            desc.Depth = depth;
            desc.Format = Format;
            desc.NumChannels = NumChannels;
            desc.Flags = flags;
            return Array(Descriptor(desc));
        }
    };
    
    /*
     cuMemcpyAto* wrapper
     */
    void Memcpy(Array &dst, const Array &src, const size_t byteCount, const size_t dstOffset, const size_t srcOffset)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes() - dstOffset, src.getTotalBytes() - srcOffset):byteCount;
        Error::Check(cuMemcpyAtoA(dst.m_array.get(), dstOffset, src.m_array.get(), srcOffset, _byteCount));
    }
    
    void Memcpy(Memory &dst, const Array &src, const size_t byteCount, const size_t srcOffset)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes(), src.getTotalBytes() - srcOffset):byteCount;
        Error::Check(cuMemcpyAtoD(*(dst.m_devptr), src.m_array.get(), srcOffset, _byteCount));
    }
    
    void Memcpy(void *dst, const Array &src, const size_t byteCount, const size_t srcOffset)
    {
        size_t _byteCount = (byteCount == 0)?(src.getTotalBytes() - srcOffset):byteCount;
        Error::Check(cuMemcpyAtoH(dst, src.m_array.get(), srcOffset, _byteCount));
    }
    
    template<typename T>
    void Memcpy(std::vector<T> &dst, const Array &src, const size_t byteCount, const size_t srcOffset)
    {
        size_t _byteCount = (byteCount == 0)?std::min(sizeof(T) * dst.size(), src.getTotalBytes() - srcOffset):byteCount;
        Error::Check(cuMemcpyAtoH(dst.data(), *(src.m_array), srcOffset, _byteCount));
    }
    
    /*
     cuMemcpyDto* wrapper
     */
    void Memcpy(Array &dst, const Memory &src, const size_t byteCount, const size_t dstOffset)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes() - dstOffset, src.getTotalBytes()):byteCount;
        Error::Check(cuMemcpyDtoA(dst.m_array.get(), dstOffset, *(src.m_devptr), _byteCount));
    }
    
    void Memcpy(Memory &dst, const Memory &src, const size_t byteCount)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes(), src.getTotalBytes()):byteCount;
        Error::Check(cuMemcpyDtoD(*(dst.m_devptr), *(src.m_devptr), _byteCount));
    }
    
    void Memcpy(void *dst, const Memory &src, const size_t byteCount)
    {
        size_t _byteCount = (byteCount == 0)?src.getTotalBytes():byteCount;
        Error::Check(cuMemcpyDtoH(dst, *(src.m_devptr), _byteCount));
    }
    
    template<typename T>
    void Memcpy(std::vector<T> &dst, const Memory &src, const size_t byteCount)
    {
        size_t _byteCount = (byteCount == 0)?std::min(sizeof(T) * dst.size(), src.getTotalBytes()):byteCount;
        Error::Check(cuMemcpyDtoH(dst.data(), *(src.m_devptr), _byteCount));
    }
    
    /*
     cuMemcpyHto* wrapper
     */
    void Memcpy(Array &dst, const void *src, const size_t byteCount, const size_t dstOffset)
    {
        size_t _byteCount = (byteCount == 0)?(dst.getTotalBytes() - dstOffset):byteCount;
        Error::Check(cuMemcpyHtoA(dst.m_array.get(), dstOffset, src, _byteCount));
    }
    
    template<typename T>
    void Memcpy(Array &dst, const std::vector<T> &src, const size_t byteCount, const size_t dstOffset)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes() - dstOffset, sizeof(T) * src.size()):byteCount;
        Error::Check(cuMemcpyHtoA(dst.m_array.get(), dstOffset, src.data(), _byteCount));
    }
    
    void Memcpy(Memory &dst, const void *src, const size_t byteCount)
    {
        size_t _byteCount = (byteCount == 0)?(dst.getTotalBytes()):byteCount;
        Error::Check(cuMemcpyHtoD(*(dst.m_devptr), src, _byteCount));
    }
    
    template<typename T>
    void Memcpy(Memory &dst, const std::vector<T> &src, const size_t byteCount)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes(), sizeof(T) * src.size()):byteCount;
        Error::Check(cuMemcpyHtoD(*(dst.m_devptr), src.data(), _byteCount));
    }
    
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
            Error::Check(cuModuleGetTexRef(&ref, mod.m_module.get(), name.c_str()));
        }
        
        ~TexRef() {}
        
        void setMemory(const Memory &mem)
        {
            Error::Check(cuTexRefSetAddress(nullptr, ref, *(mem.m_devptr), mem.getTotalBytes()));
        }
        
        void setArray(const Array &array)
        {
            Error::Check(cuTexRefSetArray(ref, array.m_array.get(), CU_TRSA_OVERRIDE_FORMAT));
        }
        
        void setAddressMode(const int dim, const CUaddress_mode mode)
        {
            /*
             線形メモリを使う場合、およびsetFlagsで CU_TRSF_NORMALIZED_COORDINATES がセットされていない場合はmodeに何を指定しても CU_TR_ADDRESS_MODE_CLAMP 固定となるので注意。
             */
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
            Error::Check(cuModuleGetFunction(&func, mod.m_module.get(), name.c_str()));
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
            args.push_back(&(*(value.m_devptr)));
            return *this;
        }
        
        template<typename T>
        Function& setArg(const unsigned int index, T &value)
        {
            if (index < args.size()) {
                args[index] = &value;
            } else {
                setArg(value);
            }
            return *this;
        }
        
        Function& setArg(const unsigned int index, Memory &value)
        {
            if (index < args.size()) {
                args[index] = &(*(value.m_devptr));
            } else {
                setArg(value);
            }
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
        friend class Timer;
        
    private:
        std::shared_ptr<CUevent_st> m_event;
        
        Event()
        : m_event()
        {
            throw "Do not use default ctor.";
        }
        
        Event(CUevent a_event)
        : m_event()
        {
            m_event = std::shared_ptr<CUevent_st>(a_event, [](CUevent_st *event){
                cuEventDestroy(event);
#ifdef DEBUG
                std::cout << "cuEventDestroy called." << std::endl;
#endif
            });
        }
        
    public:
        Event(unsigned int flags = CU_EVENT_DEFAULT)
        {
            CUevent event;
            Error::Check(cuEventCreate(&event, flags));
#ifdef DEBUG
            std::cout << "cuEventCreate called." << std::endl;
#endif
            m_event = std::shared_ptr<CUevent_st>(event, [](CUevent_st *event){
                cuEventDestroy(event);
#ifdef DEBUG
                std::cout << "cuEventDestroy called." << std::endl;
#endif
            });
        }
        
        void record(const CUstream &hStream = 0)
        {
            Error::Check(cuEventRecord(m_event.get(), hStream));
        }
        
        void query()
        {
            Error::Check(cuEventQuery(m_event.get()));
        }
        
        void synchronize()
        {
            Error::Check(cuEventSynchronize(m_event.get()));
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
            Error::Check(cuEventElapsedTime(&msec, begin.m_event.get(), end.m_event.get()));
            return msec;
        }
    };
}

#endif
