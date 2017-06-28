# TFDetect

Provides a C++ library to perform inference on models trained with the
tensorflow [object detection scripts](https://github.com/tensorflow/models/tree/master/object_detection).

## Prequesites

* OpenCV 2.4 (packages: libopencv-dev)
* Build tools (packages: build-essentials, autoconf)
* Bazel (see [here](https://bazel.build/versions/master/docs/install-ubuntu.html))

## Setup

```bash
# install c_api
cd c_api
mkdir build && build
cmake ..
make install

# compile the library
cd ../../
mkdir build && build
cmake ..
make install
```
