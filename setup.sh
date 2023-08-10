

#!/bin/bash

module swap PrgEnv-cray PrgEnv-nvhpc
module load friendly-testing
module swap nvhpc nvidia/22.11
module load cray-fftw

export CUDA_HOME=/usr/projects/hpcsoft/cos2/common/x86_64/nvidia/hpc_sdk/22.11/Linux_x86_64/22.11/cuda/11.0    # must set CUDA_HOME TO 11.0 specifically for (nvcc --version)
export CUDA_PATH=/usr/projects/hpcsoft/cos2/common/x86_64/nvidia/hpc_sdk/22.11/Linux_x86_64/22.11/cuda/11.0
export NVHPC_CUDA_HOME=/usr/projects/hpcsoft/cos2/common/x86_64/nvidia/hpc_sdk/22.11/Linux_x86_64/22.11/cuda/11.0

export XC_ROOT=$HOME/libxc-5.1.5/

#export PATH="/usr/projects/hpcsoft/cos2/common/x86_64/nvidia/hpc_sdk/22.11/Linux_x86_64/22.11/cuda/bin:$PATH"   # enable command line command (nvcc) cuda compiler
# dont have to do this if loaded modules correctly

