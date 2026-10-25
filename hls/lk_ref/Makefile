# Vitis Vision densePyrOpticalFlow C-simulation vs fcw::denseLk (needs the sparse clone in third_party/, see README.md).
#   make run PREV=prev.png CURR=curr.png
XILINX_HLS_INCLUDE ?= /mnt/c/Xilinx/Vitis/2024.2/include
VISION := ../../third_party/Vitis_Libraries/vision/L1
EX := $(VISION)/examples/lkdensepyrof
CXXFLAGS = -std=c++17 -O2 -w -DHLS_NO_XIL_FPO_LIB -I$(XILINX_HLS_INCLUDE) -I$(VISION)/include -I$(EX) -I$(EX)/config -I../../host/include $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

build/lk_ref: lk_ref.cpp ../../host/src/lk_flow.cpp ../common/fcw_pyr_dense_optical_flow.hpp
	mkdir -p build
	g++ $(CXXFLAGS) lk_ref.cpp ../../host/src/lk_flow.cpp -o $@ $(LIBS)

run: build/lk_ref
	./build/lk_ref $(PREV) $(CURR) build/vitis_flow.bin

.PHONY: run

# The FCW LK without the per-pass median filters (hls/common/fcw_pyr_dense_optical_flow.hpp, LK_MEDIAN=0).
build/lk_ref_nomed: lk_ref.cpp ../../host/src/lk_flow.cpp ../common/fcw_pyr_dense_optical_flow.hpp
	mkdir -p build
	g++ $(CXXFLAGS) -DLK_MEDIAN=0 lk_ref.cpp ../../host/src/lk_flow.cpp -o $@ $(LIBS)

# Median off, a smaller remap line buffer: build/lk_ref_nomed_l26, build/lk_ref_nomed_l34, ...
build/lk_ref_nomed_l%: lk_ref.cpp ../../host/src/lk_flow.cpp ../common/fcw_pyr_dense_optical_flow.hpp
	mkdir -p build
	g++ $(CXXFLAGS) -DLK_MEDIAN=0 -DLK_LINES=$* lk_ref.cpp ../../host/src/lk_flow.cpp -o $@ $(LIBS)

# 3x3 median instead of AMD's 5x5: build/lk_ref_med3
build/lk_ref_med3: lk_ref.cpp ../../host/src/lk_flow.cpp ../common/fcw_pyr_dense_optical_flow.hpp
	mkdir -p build
	g++ $(CXXFLAGS) -DLK_MEDIAN=3 lk_ref.cpp ../../host/src/lk_flow.cpp -o $@ $(LIBS)

# 3x3 median with a smaller remap line buffer: build/lk_ref_med3_l34, ...
build/lk_ref_med3_l%: lk_ref.cpp ../../host/src/lk_flow.cpp ../common/fcw_pyr_dense_optical_flow.hpp
	mkdir -p build
	g++ $(CXXFLAGS) -DLK_MEDIAN=3 -DLK_LINES=$* lk_ref.cpp ../../host/src/lk_flow.cpp -o $@ $(LIBS)
