# Vitis Vision densePyrOpticalFlow C-simulation vs fcw::denseLk (needs the sparse clone in third_party/, see README.md).
#   make run PREV=prev.png CURR=curr.png
XILINX_HLS_INCLUDE ?= /mnt/c/Xilinx/Vitis/2024.2/include
VISION := ../../third_party/Vitis_Libraries/vision/L1
EX := $(VISION)/examples/lkdensepyrof
CXXFLAGS = -std=c++17 -O2 -w -DHLS_NO_XIL_FPO_LIB -I$(XILINX_HLS_INCLUDE) -I$(VISION)/include -I$(EX) -I$(EX)/config -I../../host/include $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

build/lk_ref: lk_ref.cpp ../../host/src/lk_flow.cpp
	mkdir -p build
	g++ $(CXXFLAGS) lk_ref.cpp ../../host/src/lk_flow.cpp -o $@ $(LIBS)

run: build/lk_ref
	./build/lk_ref $(PREV) $(CURR) build/vitis_flow.bin

.PHONY: run
