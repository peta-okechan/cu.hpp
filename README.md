cu.hpp
======

The C++ Wrapper for CUDA Driver API

OpenCLのC++ Wrapperであるcl.hppにインスパイアされて作成しました。
cl.hppと比べて設計も実装もかなりしょぼいです。
現在CUDA Driver APIのごく一部にのみ対応しています。
未対応のAPIについては個人的に必要になったときに随時追加していく予定です。

cu.hppのライセンスはMITライセンスとなってます。

今のところ動作確認した環境は以下のとおりです。

*OS*
* Mac OSX 10.8.3
* Ubuntu 13.04
* Windows 7

*コンパイラ*
* Clang(LLVM 3.1 and 3.2, C++11)

*CUDA API*
* CUDA 5.0 and 5.5 Driver API

基本的にC++11に対応したコンパイラ & CUDA 5.0以上なら動作すると思われますが、上記の環境以外については未確認です。

使い方についてはexampleディレクトリのmain.cppを参照。