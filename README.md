# TFDetect

Provides a C++ library to perform inference on models trained with the
tensorflow [object detection scripts](https://github.com/tensorflow/models/tree/master/object_detection).

## Acknowledgments

The CMake scripts to build the tensorflow C-API are based on the CMake scripts for the C++ API from <https://github.com/FloopCZ/tensorflow_cc>.

## Prequesites

* OpenCV 2.4 (packages: libopencv-dev)
* Build tools (packages: build-essentials, autoconf)
* Bazel (see [here](https://bazel.build/versions/master/docs/install-ubuntu.html))

## Setup

```bash
# install c_api
cd c_api
mkdir build && cd build
cmake ..
# set TF_NEED_CUDA=1 for GPU support.
# If you do so, you will be prompted to provide some information
# about your CUDA installation after some time.
TF_NEED_CUDA=0 make install

# compile the library
cd ../../
mkdir build && cd build
cmake ..
make install
```
