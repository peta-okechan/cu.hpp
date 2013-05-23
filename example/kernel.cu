//
//  kernel.cu
//  example
//
//  Created by peta on 2013/05/23.
//  Copyright (c) 2013年 peta.okechan.net. All rights reserved.
//

extern "C"
{
    __global__ void addone(float *v, int n)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;

        if (index < n) {
            v[index] += 1.0f;
        }
    }
}
