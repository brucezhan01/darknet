COMMON_REPO := ./
#Common Includes
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

darknet_SRCS=./src/gemm.cpp
darknet_SRCS+=./src/utils.cpp
darknet_SRCS+=./src/cuda.cpp
darknet_SRCS+=./src/deconvolutional_layer.cpp
darknet_SRCS+=./src/convolutional_layer.cpp
darknet_SRCS+=./src/list.cpp
darknet_SRCS+=./src/image.cpp
darknet_SRCS+=./src/activations.cpp
darknet_SRCS+=./src/im2col.cpp
darknet_SRCS+=./src/col2im.cpp
darknet_SRCS+=./src/blas.cpp
darknet_SRCS+=./src/crop_layer.cpp
darknet_SRCS+=./src/dropout_layer.cpp
darknet_SRCS+=./src/maxpool_layer.cpp
darknet_SRCS+=./src/softmax_layer.cpp
darknet_SRCS+=./src/data.cpp
darknet_SRCS+=./src/matrix.cpp
darknet_SRCS+=./src/network.cpp
darknet_SRCS+=./src/connected_layer.cpp
darknet_SRCS+=./src/cost_layer.cpp
darknet_SRCS+=./src/parser.cpp
darknet_SRCS+=./src/option_list.cpp
darknet_SRCS+=./src/detection_layer.cpp
darknet_SRCS+=./src/captcha.cpp
darknet_SRCS+=./src/route_layer.cpp
darknet_SRCS+=./src/writing.cpp
darknet_SRCS+=./src/box.cpp
darknet_SRCS+=./src/nightmare.cpp
darknet_SRCS+=./src/normalization_layer.cpp
darknet_SRCS+=./src/avgpool_layer.cpp
darknet_SRCS+=./src/coco.cpp
darknet_SRCS+=./src/dice.cpp
darknet_SRCS+=./src/yolo.cpp
darknet_SRCS+=./src/detector.cpp
darknet_SRCS+=./src/layer.cpp
darknet_SRCS+=./src/compare.cpp
darknet_SRCS+=./src/regressor.cpp
darknet_SRCS+=./src/classifier.cpp
darknet_SRCS+=./src/local_layer.cpp
darknet_SRCS+=./src/swag.cpp
darknet_SRCS+=./src/shortcut_layer.cpp
darknet_SRCS+=./src/activation_layer.cpp
darknet_SRCS+=./src/rnn_layer.cpp
darknet_SRCS+=./src/gru_layer.cpp
darknet_SRCS+=./src/rnn.cpp
darknet_SRCS+=./src/rnn_vid.cpp
darknet_SRCS+=./src/crnn_layer.cpp
darknet_SRCS+=./src/demo.cpp
darknet_SRCS+=./src/tag.cpp
darknet_SRCS+=./src/cifar.cpp
darknet_SRCS+=./src/go.cpp
darknet_SRCS+=./src/batchnorm_layer.cpp
darknet_SRCS+=./src/art.cpp
darknet_SRCS+=./src/region_layer.cpp
darknet_SRCS+=./src/reorg_layer.cpp
darknet_SRCS+=./src/lsd.cpp
darknet_SRCS+=./src/super.cpp
darknet_SRCS+=./src/voxel.cpp
darknet_SRCS+=./src/tree.cpp
darknet_SRCS+=./src/darknet.cpp

darknet_CXXFLAGS=-Wno-write-strings -std=c++0x -Wall -Wfatal-errors -Ofast

# Host Application
host_SRCS=$(darknet_SRCS) $(xcl2_SRCS)
host_HDRS=$(xcl2_HDRS)
host_CXXFLAGS=-I./src/ $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS) $(darknet_CXXFLAGS)
host_LDFLAGS=$(opencl_LDFLAGS)

EXES=host

# Kernel
vadd_SRCS=./fpga_src/vadd.cl
vadd_CLFLAGS=-k vadd

XOS=vadd

vadd_XOS=vadd

XCLBINS=vadd

# check
check_EXE=host
check_XCLBINS=vadd

CHECKS=check

include $(COMMON_REPO)/utility/rules.mk

