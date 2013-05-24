//
//  kernel.cu
//  example
//
//  Created by peta on 2013/05/23.
//  Copyright (c) 2013年 peta.okechan.net. All rights reserved.
//

texture<float, cudaTextureType1D, cudaReadModeElementType> twos;
__device__ float ones[10];

extern "C"
{
    __global__ void addone(float *v, int n)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;

        if (index < n) {
            v[index] += tex1D(twos, float(index) / 9.9999f) - ones[index % 10];
        }
    }
}
