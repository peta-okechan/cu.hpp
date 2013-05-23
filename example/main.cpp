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
        
        // ドライババージョンの表示
        std::cout << "CUDA driver version: " << cu::GetDriverVersion() << "\n";
        
        // CUDAデバイスを取得
        auto devices = cu::Device::GetDevices();
        if (devices.size() == 0) {
            printf("Error: No CUDA Device.\n");
            return EXIT_FAILURE;
        }
        
        // デバイス名の表示
        std::cout << devices.size() << " device(s) found.\n";
        for (auto d : devices) {
            std::cout << "  Name: " << d.getName() << ", Memory: " << d.getTotalMemBytes() / 1024 / 1024 << " MB\n";
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
        
        // 処理時間を計測
        cu::Timer timer;
        timer.start();
        
        // カーネルの実行
        addone.setArg(dData).setArg(n);
        addone.launchKernel(gridDim, blockDim);
        
        // 処理時間を表示
        timer.stop();
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout << "Kernel execution time: " << timer.elapsedMilliSec() / 1000.0f << " sec.\n";
        std::cout.setf(std::ios_base::fmtflags(0), std::ios_base::floatfield);
        
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
        std::cout << e.what() << "\n";
    }
    
    return EXIT_FAILURE;
}