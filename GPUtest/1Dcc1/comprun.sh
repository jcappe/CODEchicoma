
nvcc -x cu main.cpp -o main.exe -I/usr/projects/hpcsoft/cos2/common/x86_64/nvidia/hpc_sdk/22.11/Linux_x86_64/22.11/compilers/bin/include -I../utils -L/usr/projects/hpcsoft/cos2/common/x86_64/nvidia/hpc_sdk/22.11/Linux_x86_64/22.11/math_libs/lib64 -lcufft -O3 -std=c++17
./main.exe