# Project Title

A simple C++ dnn demo, based on tensorflow, run with qtcreator. cuda supported.

## Getting Started

Because of the struggling reading and debugging with python dnn code, I found libtensorflow_cc.so and libtensorflow_framework.so
have good interfaces to be used by C++ project. The goal is to build a maintainable and debugable c++ project for an easy used dnn net.

### Prerequisites

1.Tensorflow r1.3+
2.Cuda, must have /usr/local/cuda
3.Built your tensorflow for c++

```
For example, you can build your tensorflow for python first
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package

Then don't forget to build it for c++
bazel build --config=cuda //tensorflow:libtensorflow_cc.so
```

### Run in qtcreator

In the project root path, run configure and give the infomations of your tensorflow
```
./configure
# Your tensorflow version (default 1.3):
1.3
  Project will be built based on your tensorflow r1.3

# Your bazel-tensorflow path (default /mnt/1t/AI/tensorflow/bazel-tensorflow):
/mnt/1t/AI/tensorflow/bazel-tensorflow
  Set bazel-tensorflow path as /mnt/1t/AI/tensorflow/bazel-tensorflow
```

Then open your qtcreator, open the cnaidnn.pro as importing the project into your IDE, build and run in the qtcreator.

## Running the tests

The tests/test01.cpp is defaultly built in the project as the main function. It is a very tiny test which
gives 4 floats as inputs and 4 as outputs. The 4 outputs marking which input is largest will be 1 rather than 0.

## License
Free to go.
