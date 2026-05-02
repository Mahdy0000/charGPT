@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%
mkdir build
nvcc -O3 -arch=sm_86 src\main.cpp src\cuda_kernels.cu -o build\llm.exe -lcublas
nvcc -O3 -arch=sm_86 src\chat.cpp src\cuda_kernels.cu -o build\chat.exe -lcublas
