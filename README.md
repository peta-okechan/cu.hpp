cu.hpp
======

The C++ Wrapper for CUDA Driver API

OpenCLのC++ Wrapperであるcl.hppにインスパイアされて作成しました。
cl.hppと比べて設計も実装もかなりしょぼいです。
現在CUDA Driver APIのごく一部にのみ対応しています。
未対応のAPIについては個人的に必要になったときに随時追加していく予定です。

cu.hppのライセンスはMITライセンスとなってます。

現在の対応環境は、Mac OSX 10.8.3 & clang(C++11) & CUDA 5.0となっています。
これ以外の環境では動かないかもしれません。（動くかもしれません。）

使い方についてはexampleディレクトリのmain.cppを参照。