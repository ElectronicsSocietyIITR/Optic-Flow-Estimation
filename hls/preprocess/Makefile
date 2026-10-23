# Preprocessing kernel, C-simulation with g++ and the Vitis HLS headers (needs third_party/Vitis_Libraries, see hls/lk_ref/README.md).
#   make csim [FRAME=dashcam.png]
XILINX_HLS_INCLUDE ?= /mnt/c/Xilinx/Vitis/2024.2/include
VISION := ../../third_party/Vitis_Libraries/vision/L1/include
CXXFLAGS = -std=c++17 -O2 -w -DHLS_NO_XIL_FPO_LIB -I$(XILINX_HLS_INCLUDE) -I$(VISION) $(shell pkg-config --cflags opencv4)

csim: build/csim
	./build/csim $(FRAME)

build/csim: src/preprocess.cpp src/preprocess.hpp tb/tb_preprocess.cpp
	mkdir -p build
	g++ $(CXXFLAGS) src/preprocess.cpp tb/tb_preprocess.cpp -o $@ $(shell pkg-config --libs opencv4)

.PHONY: csim
