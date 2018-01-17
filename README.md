# TFDetect

Provides a C++ library to perform inference on models trained with the
tensorflow [object detection scripts](https://github.com/tensorflow/models/tree/master/object_detection).

## Prequesites

* OpenCV 2.4 (packages: libopencv-dev)
* Build tools (packages: build-essentials, autoconf)
* Bazel (see [here](https://bazel.build/versions/master/docs/install-ubuntu.html))

## Setup

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<FILL_IN> ..
# bazel opts are needed for bazel 9.0
ADDITIONAL_BAZEL_OPTS='--incompatible_load_argument_is_label=false' TF_NEED_MKL=1 make
# you can also build with CUDA support. Note that you will be prompted for CUDA specific configurations during the build
ADDITIONAL_BAZEL_OPTS='--incompatible_load_argument_is_label=false' TF_NEED_MKL=1 TF_NEED_CUDA=1 make
make install
```
