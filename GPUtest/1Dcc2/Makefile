
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))
INC          := -I$(CUDA_TOOLKIT)/../include -I../utils
LIBS         := -L$(CUDA_TOOLKIT)/../../math_libs/lib64 -lcufft
FLAGS        := -O3 -std=c++17

default: main.cpp
	nvcc -x cu $(FLAGS) $(INC) $(LIBS) main.cpp -o main.exe