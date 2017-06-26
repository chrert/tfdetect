#!/bin/sh
set -e

# configure environmental variables
export CC_OPT_FLAGS="-march=native"
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_NEED_JEMALLOC=1
export TF_NEED_VERBS=0
export TF_NEED_MKL=1
export TF_DOWNLOAD_MKL=1
export TF_NEED_MPI=0
export TF_ENABLE_XLA=1
export TF_CUDA_CLANG=0
export TF_NEED_CUDA=0

export PYTHON_BIN_PATH="$(which python)"
export PYTHON_LIB_PATH="$(${PYTHON_BIN_PATH} -c 'import site; print(site.getsitepackages()[0])')"


# configure and build
./configure
bazel build --config=opt --copt=${CC_OPT_FLAGS} tensorflow:libtensorflow.so
bazel shutdown
