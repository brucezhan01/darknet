#   Buils default project, do not run synthesis
#     vivado_hls -f run-hls.tcl 
#     vivado_hls -p prj_hls_darknet &

array set opt {
  doCsim      0
  doRTLsynth  1
  doRTLsim    0
}

foreach arg $::argv {
  #puts "ARG $arg"
  foreach o [lsort [array names opt]] {
    if {[regexp "$o +(\\w+)" $arg unused opt($o)]} {
      puts "  Setting CONFIG  $o  [set opt($o)]"
    }
  }
}

puts "Final CONFIG"

set proj_dir "prj_hls_darknet"
open_project $proj_dir -reset
#set_top Kernel_0
set CFLAGS_H "-I../src -Wno-write-strings -std=c++0x -Wall -Wfatal-errors  -Ofast"
add_files  -tb ./src/gemm.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/utils.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/cuda.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/deconvolutional_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/convolutional_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/list.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/image.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/activations.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/im2col.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/col2im.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/blas.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/crop_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/dropout_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/maxpool_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/softmax_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/data.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/matrix.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/network.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/connected_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/cost_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/parser.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/option_list.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/detection_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/captcha.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/route_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/writing.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/box.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/nightmare.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/normalization_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/avgpool_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/coco.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/dice.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/yolo.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/detector.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/compare.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/regressor.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/classifier.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/local_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/swag.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/shortcut_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/activation_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/rnn_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/gru_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/rnn.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/rnn_vid.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/crnn_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/demo.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/tag.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/cifar.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/go.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/batchnorm_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/art.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/region_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/reorg_layer.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/lsd.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/super.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/voxel.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/tree.cpp    -cflags "$CFLAGS_H"
add_files  -tb ./src/darknet.cpp    -cflags "$CFLAGS_H"

open_solution "solution1"
config_compile -ignore_long_run_time
#config_schedule -effort medium -verbose

set_part {xcku115-flvb2104-2-e} -tool vivado

create_clock -period 3.333333 -name default

set run_args "detector test ./cfg/voc.data ./cfg/tiny-yolo-voc.cfg ./tiny-yolo-voc.weights ./data/dog.jpg"

if {$opt(doCsim)} {
  puts "***** C SIMULATION *****"
  csim_design -ldflags "-lz -lrt -lstdc++" -argv "$run_args"
}

if {$opt(doRTLsynth)} {
  puts "***** C/RTL SYNTHESIS *****"
  csynth_design
  if {$opt(doRTLsim)} {
    puts "***** C/RTL SIMULATION *****"
    cosim_design -trace_level all -ldflags "-lrt" -argv "$run_args"
  }
}

if {$opt(doExport)} {
    export_design -evaluate verilog -format ipxact
}

quit
