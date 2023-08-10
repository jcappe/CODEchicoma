set -x
NVHPC_VER=22.11
CUDA_VER=11.8

# NVHPC_VER=23.1
# CUDA_VER=12.0

$NVHPC_ROOT/comm_libs/mpi/bin/mpic++ -std=c++17 -I$NVHPC_ROOT/math_libs/$CUDA_VER/include/cufftmp/ \
-I$NVHPC_ROOT/cuda/$CUDA_VER/include/ of.cu -L$NVHPC_ROOT/math_libs/$CUDA_VER/lib64/ \
-L$NVHPC_ROOT/cuda/$CUDA_VER/lib64/ -lcudart -lcufftMp -o of.x \
# -gpu=cc80
# -gencode arch=compute_XX,code=sm_70