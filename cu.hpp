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
#include <unordered_map>
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
     Resource release templates. 
     */
    template<typename T>
    void Release(const T value)
    {
        throw Error(CUDA_ERROR_UNKNOWN, "No releasable type.");
    }
    
    template<>
    void Release<CUcontext>(const CUcontext context)
    {
        Error::Check(cuCtxDestroy(context));
    }
    
    template<>
    void Release<CUmodule>(const CUmodule mod)
    {
        Error::Check(cuModuleUnload(mod));
    }
    
    template<>
    void Release<CUdeviceptr>(const CUdeviceptr ptr)
    {
        Error::Check(cuMemFree(ptr));
    }
    
    template<>
    void Release<CUarray>(const CUarray handle)
    {
        Error::Check(cuArrayDestroy(handle));
    }
    
    template<>
    void Release<CUevent>(const CUevent e)
    {
        Error::Check(cuEventDestroy(e));
    }
    
    /*
     CUDA Resource Manager
     */
    template<typename T>
    class ResourceManager
    {
    private:
        std::unordered_map<T, unsigned int> refCount;
    public:
        static ResourceManager* GetInstance()
        {
            static ResourceManager<T> *m = nullptr;
            if (!m) m = new ResourceManager<T>();
            return m;
        }
        
        ResourceManager() {}
        
        ~ResourceManager() {}
        
        unsigned int retain(const T key)
        {
            refCount[key] += 1;
            return refCount[key];
        }
        
        unsigned int release(const T key)
        {
            if (refCount[key] > 0) {
                refCount[key] -= 1;
                if (refCount[key] == 0) {
                    destroy(key);
                }
            }
            return refCount[key];
        }
        
        virtual void destroy(const T key)
        {
            Release<T>(key);
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
        typedef ResourceManager<CUcontext> Manager;
    private:
        CUcontext ctx;
        
        Context(const CUcontext &_ctx)
        : ctx(_ctx)
        {
            Manager::GetInstance()->retain(ctx);
        }
        
    public:
        Context(Device device, unsigned int flags = CU_CTX_SCHED_AUTO)
        {
            Error::Check(cuCtxCreate(&ctx, flags, device()));
            Manager::GetInstance()->retain(ctx);
        }
        
        Context(const Context& _context)
        : Context(_context())
        {}
        
        ~Context()
        {
            Manager::GetInstance()->release(ctx);
        }
        
        bool operator==(const Context &rhs) const
        {
            return (ctx == rhs());
        }
        
        CUcontext operator()(void) const
        {
            return ctx;
        }
        
        CUfunc_cache getCacheConfig()
        {
            CUfunc_cache config;
            Context::push(*this);
            Error::Check(cuCtxGetCacheConfig(&config));
            Context::pop();
            return config;
        }
        
        Device getDevice()
        {
            CUdevice dev;
            Context::push(*this);
            Error::Check(cuCtxGetDevice(&dev));
            Context::pop();
            return Device(dev);
        }
        
        template<CUlimit limit>
        size_t getLimit()
        {
            size_t pvalue;
            Context::push(*this);
            Error::Check(cuCtxGetLimit(&pvalue, limit));
            Context::pop();
            return pvalue;
        }
        
        CUsharedconfig getSharedMemConfig()
        {
            CUsharedconfig pconfig;
            Context::push(*this);
            Error::Check(cuCtxGetSharedMemConfig(&pconfig));
            Context::pop();
            return pconfig;
        }
        
        void setCacheConfig(CUfunc_cache config)
        {
            Context::push(*this);
            Error::Check(cuCtxSetCacheConfig(config));
            Context::pop();
        }
        
        template<CUlimit limit>
        void setLimit(size_t value)
        {
            Context::push(*this);
            Error::Check(cuCtxSetLimit(limit, value));
            Context::pop();
        }
        
        void getSharedMemConfig(CUsharedconfig config)
        {
            Context::push(*this);
            Error::Check(cuCtxSetSharedMemConfig(config));
            Context::pop();
        }
        
        void synchronize()
        {
            Context::push(*this);
            Error::Check(cuCtxSynchronize());
            Context::pop();
        }
        
        unsigned int getApiVersion()
        {
            unsigned int version;
            Error::Check(cuCtxGetApiVersion(ctx, &version));
            return version;
        }
        
        void getMemInfo(size_t &memFree, size_t &memTotal) const
        {
            Context::push(*this);
            Error::Check(cuMemGetInfo(&memFree, &memTotal));
            Context::pop();
        }
        
        static void push(const Context &context)
        {
            Error::Check(cuCtxPushCurrent(context()));
        }
        
        static Context pop()
        {
            CUcontext ret;
            Error::Check(cuCtxPopCurrent(&ret));
            return Context(ret);
        }
        
        static Context getCurrent()
        {
            CUcontext ctx;
            Error::Check(cuCtxGetCurrent(&ctx));
            return Context(ctx);
        }
        
        static void setCurrent(const Context &_context)
        {
            Error::Check(cuCtxSetCurrent(_context()));
        }
    };
    
    /*
     CUmodule wrapper
     */
    class Module
    {
        typedef ResourceManager<CUmodule> Manager;
    private:
        CUmodule mod;
        
        Module(const CUmodule _mod)
        : mod(_mod)
        {
            Manager::GetInstance()->retain(mod);
        }
    public:        
        ~Module()
        {
            Manager::GetInstance()->release(mod);
        }
        
        CUmodule operator()(void) const
        {
            return mod;
        }
        
        static Module LoadFromFile(const std::string modulePath)
        {
            CUmodule mod;
            Error::Check(cuModuleLoad(&mod, modulePath.c_str()));
            return Module(mod);
        }
        
        static Module LoadFromImage(const void* image)
        {
            CUmodule mod;
            Error::Check(cuModuleLoadData(&mod, image));
            return Module(mod);
        }
        
        static Module LoadFromFatBinary(const void* fatCubin)
        {
            CUmodule mod;
            Error::Check(cuModuleLoadFatBinary(&mod, fatCubin));
            return Module(mod);
        }
    };
    
    /*
     CUdeviceptr wrapper
     */
    class Memory
    {
        typedef ResourceManager<CUdeviceptr> Manager;
    private:
        CUdeviceptr ptr;
        size_t byteCount;
    public:
        Memory(const size_t _byteCount)
        : byteCount(_byteCount)
        {
            Error::Check(cuMemAlloc(&ptr, byteCount));
            Manager::GetInstance()->retain(ptr);
        }
        
        Memory(const Module &mod, const std::string name)
        {
            Error::Check(cuModuleGetGlobal(&ptr, &byteCount, mod(), name.c_str()));
            /*
             cuMemAlloc() と cuMemAllocPitch() 以外で取得した CUdeviceptr は
             cuMemFree() で解放する必要はない（解放できない）ので
             リソースマネージャには登録しない。
             */
        }
        
        ~Memory()
        {
            Manager::GetInstance()->release(ptr);
        }
        
        CUdeviceptr operator()(void) const
        {
            return ptr;
        }
        
        CUdeviceptr* operator()(void)
        {
            return &ptr;
        }
        
        size_t getTotalBytes(void) const
        {
            return byteCount;
        }
    };
    
    /*
     CUarray wrapper
     */
    class Array
    {
        typedef ResourceManager<CUarray> Manager;
    private:
        class Descriptor
        {
        private:
            bool is3D;
            size_t width, height, depth;
            CUarray_format format;
            unsigned int numChannels, flags;
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
            
            void arrayCreate(CUarray &handle)
            {
                if (is3D) {
                    CUDA_ARRAY3D_DESCRIPTOR desc;
                    getDescriptor(desc);
                    Error::Check(cuArray3DCreate(&handle, &desc));
                } else {
                    CUDA_ARRAY_DESCRIPTOR desc;
                    getDescriptor(desc);
                    Error::Check(cuArrayCreate(&handle, &desc));
                }
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
        
        CUarray handle;
        Descriptor desc;
    public:
        Array(const Descriptor &_desc)
        : desc(_desc)
        {
            desc.arrayCreate(handle);
            Manager::GetInstance()->retain(handle);
        }
        
        ~Array()
        {
            Manager::GetInstance()->release(handle);
        }
        
        CUarray operator()(void) const
        {
            return handle;
        }
        
        CUarray* operator()(void)
        {
            return &handle;
        }
        
        Descriptor getDescriptor(const bool useCache = true)
        {
            if (!useCache) {
                desc.update(handle);
            }
            return desc;
        }
        
        size_t getTotalBytes() const
        {
            return desc.getTotalBytes();
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
    void Memcpy(Array &dst, const Array &src, const size_t byteCount = 0, const size_t dstOffset = 0, const size_t srcOffset = 0)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes() - dstOffset, src.getTotalBytes() - srcOffset):byteCount;
        Error::Check(cuMemcpyAtoA(*dst(), dstOffset, src(), srcOffset, _byteCount));
    }
    
    void Memcpy(Memory &dst, const Array &src, const size_t byteCount = 0, const size_t srcOffset = 0)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes(), src.getTotalBytes() - srcOffset):byteCount;
        Error::Check(cuMemcpyAtoD(*dst(), src(), srcOffset, _byteCount));
    }
    
    void Memcpy(void *dst, const Array &src, const size_t byteCount = 0, const size_t srcOffset = 0)
    {
        size_t _byteCount = (byteCount == 0)?(src.getTotalBytes() - srcOffset):byteCount;
        Error::Check(cuMemcpyAtoH(dst, src(), srcOffset, _byteCount));
    }
    
    template<typename T>
    void Memcpy(std::vector<T> &dst, const Array &src, const size_t byteCount = 0, const size_t srcOffset = 0)
    {
        size_t _byteCount = (byteCount == 0)?std::min(sizeof(T) * dst.size(), src.getTotalBytes() - srcOffset):byteCount;
        Error::Check(cuMemcpyAtoH(dst.data(), src(), srcOffset, _byteCount));
    }
    
    /*
     cuMemcpyDto* wrapper
     */
    void Memcpy(Array &dst, const Memory &src, const size_t byteCount = 0, const size_t dstOffset = 0)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes() - dstOffset, src.getTotalBytes()):byteCount;
        Error::Check(cuMemcpyDtoA(*dst(), dstOffset, src(), _byteCount));
    }
    
    void Memcpy(Memory &dst, const Memory &src, const size_t byteCount = 0)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes(), src.getTotalBytes()):byteCount;
        Error::Check(cuMemcpyDtoD(*dst(), src(), _byteCount));
    }
    
    void Memcpy(void *dst, const Memory &src, const size_t byteCount = 0)
    {
        size_t _byteCount = (byteCount == 0)?src.getTotalBytes():byteCount;
        Error::Check(cuMemcpyDtoH(dst, src(), _byteCount));
    }
    
    template<typename T>
    void Memcpy(std::vector<T> &dst, const Memory &src, const size_t byteCount = 0)
    {
        size_t _byteCount = (byteCount == 0)?std::min(sizeof(T) * dst.size(), src.getTotalBytes()):byteCount;
        Error::Check(cuMemcpyDtoH(dst.data(), src(), _byteCount));
    }
    
    /*
     cuMemcpyHto* wrapper
     */
    void Memcpy(Array &dst, const void *src, const size_t byteCount = 0, const size_t dstOffset = 0)
    {
        size_t _byteCount = (byteCount == 0)?(dst.getTotalBytes() - dstOffset):byteCount;
        Error::Check(cuMemcpyHtoA(*dst(), dstOffset, src, _byteCount));
    }
    
    template<typename T>
    void Memcpy(Array &dst, const std::vector<T> &src, const size_t byteCount = 0, const size_t dstOffset = 0)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes() - dstOffset, sizeof(T) * src.size()):byteCount;
        Error::Check(cuMemcpyHtoA(*dst(), dstOffset, src.data(), _byteCount));
    }
    
    void Memcpy(Memory &dst, const void *src, const size_t byteCount = 0)
    {
        size_t _byteCount = (byteCount == 0)?(dst.getTotalBytes()):byteCount;
        Error::Check(cuMemcpyHtoD(*dst(), src, _byteCount));
    }
    
    template<typename T>
    void Memcpy(Memory &dst, const std::vector<T> &src, const size_t byteCount = 0)
    {
        size_t _byteCount = (byteCount == 0)?std::min(dst.getTotalBytes(), sizeof(T) * src.size()):byteCount;
        Error::Check(cuMemcpyHtoD(*dst(), src.data(), _byteCount));
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
            Error::Check(cuModuleGetTexRef(&ref, mod(), name.c_str()));
        }
        
        ~TexRef() {}
        
        void setMemory(const Memory &mem)
        {
            Error::Check(cuTexRefSetAddress(nullptr, ref, mem(), mem.getTotalBytes()));
        }
        
        void setArray(const Array &array)
        {
            Error::Check(cuTexRefSetArray(ref, array(), CU_TRSA_OVERRIDE_FORMAT));
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
        typedef ResourceManager<CUevent> Manager;
    private:
        CUevent e;
    public:
        Event(unsigned int flags = CU_EVENT_DEFAULT)
        {
            Error::Check(cuEventCreate(&e, flags));
            Manager::GetInstance()->retain(e);
        }
        
        ~Event()
        {
            Manager::GetInstance()->release(e);
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
