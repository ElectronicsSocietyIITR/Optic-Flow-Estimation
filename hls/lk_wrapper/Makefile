# One-start dense LK, C-simulation against hls/lk_ref (needs third_party/Vitis_Libraries, see hls/lk_ref/README.md).
#   make csim PREV=prev.png CURR=curr.png
XILINX_HLS_INCLUDE ?= /mnt/c/Xilinx/Vitis/2024.2/include
VISION := ../../third_party/Vitis_Libraries/vision/L1/include
PREV ?= ../../host/tests/golden_local/lk_prev.png
CURR ?= ../../host/tests/golden_local/lk_curr.png
CXXFLAGS = -std=c++17 -O2 -w -DHLS_NO_XIL_FPO_LIB -I$(XILINX_HLS_INCLUDE) -I$(VISION) $(shell pkg-config --cflags opencv4)

csim: build/csim
	$(MAKE) -C ../lk_ref build/lk_ref
	../lk_ref/build/lk_ref $(PREV) $(CURR) build/reference_flow.bin
	./build/csim $(PREV) $(CURR) build/reference_flow.bin

build/csim: src/lk_wrapper.cpp src/lk_wrapper.hpp tb/tb_lk_wrapper.cpp
	mkdir -p build
	g++ $(CXXFLAGS) src/lk_wrapper.cpp tb/tb_lk_wrapper.cpp -o $@ $(shell pkg-config --libs opencv4)

.PHONY: csim
