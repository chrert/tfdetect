# TFDetect

Provides a C++ library to perform inference on models trained with the
tensorflow [object detection scripts](https://github.com/tensorflow/models/tree/master/object_detection).

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
