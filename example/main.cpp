//
//  main.cpp
//  example
//
//  Created by peta on 2013/05/23.
//  Copyright (c) 2013年 peta.okechan.net. All rights reserved.
//

#include <iostream>
#include "cu.hpp"

int main(int argc, const char * argv[])
{
    // データ数
    int n = 1000;
    
    // 元データの作成
    std::vector<float> hData;
    hData.reserve(n);
    for (int i = 0; i < n; i++) {
        hData.push_back(float(i));
    }
    
    // グリッドの分割数、ブロックの分割数
    int threadsPerBlock = 32;
    cu::Dim3 gridDim((n + threadsPerBlock - 1) / threadsPerBlock, 1, 1);
    cu::Dim3 blockDim(threadsPerBlock, 1, 1);
    
    try {
        // CUDAの初期化
        cu::Init();
        
        // CUDAデバイスを取得
        auto devices = cu::Device::GetDevices();
        if (devices.size() == 0) {
            printf("Error: No CUDA Device.\n");
            return EXIT_FAILURE;
        }
        
        // コンテキストの作成
        cu::Context context(devices[0]);
        
        // メモリ利用状況の表示
        size_t memFree, memTotal;
        context.getMemInfo(memFree, memTotal);
        std::cout << "Memory: total " << memTotal / 1024 / 1024 << "MB, free " << memFree / 1024 / 1024 << "MB\n";
        
        // カーネルのロード
        cu::Module mod("kernel.ptx");
        
        // カーネル関数の取得
        cu::Function addone(mod, "addone");
        
        // デバイスメモリにデータ領域を確保してホストからデータをコピー
        cu::Memory dData(sizeof(float) * hData.size());
        dData.copyFrom(hData);
        
        // カーネルの実行
        addone.setArg(dData).setArg(n);
        addone.launchKernel(gridDim, blockDim);
        
        // 結果をデバイスからホストへコピー
        dData.copyTo(hData);
        
        // 結果の表示
        std::cout << "Result: " << n << " elements\n";
        for (int i = 0; i < n; i++) {
            std::cout << hData[i] << ", ";
        }
        std::cout << "\nDone.\n";
        
        return EXIT_SUCCESS;
        
    } catch (cu::Error e) {
        std::cout << e.string() << "\n";
    }
    
    return EXIT_FAILURE;
}