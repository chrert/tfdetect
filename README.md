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
make install

# compile the library
cd ../../
mkdir build && cd build
cmake ..
make install
```
