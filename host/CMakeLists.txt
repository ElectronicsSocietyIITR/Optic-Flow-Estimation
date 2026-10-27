cmake_minimum_required(VERSION 3.22)
project(fcw_host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FCW_BOARD "Build the KV260 accelerator backends (XRT, VART); board image only" OFF)
option(FCW_BOARD_IO "Build the board's capture and display backends (GStreamer, libdrm) where available" ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d video videoio highgui)
find_package(Threads REQUIRED)

file(GLOB FCW_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp)
list(FILTER FCW_SOURCES EXCLUDE REGEX ".*/app_main\\.cpp$")
add_library(fcw STATIC ${FCW_SOURCES})
target_include_directories(fcw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fcw PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_options(fcw PRIVATE -Wall -Wextra)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/app_main.cpp)
  add_executable(fcw_app ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/app_main.cpp)
  target_link_libraries(fcw_app PRIVATE fcw)
endif()

# CPU-only baseline (no FPGA, no DPU) for the latency and energy comparison
add_executable(fcw_cpu_baseline ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpu_baseline.cpp)
target_link_libraries(fcw_cpu_baseline PRIVATE fcw)

enable_testing()
find_package(GTest REQUIRED)
file(GLOB FCW_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
add_executable(fcw_tests ${FCW_TESTS})
target_link_libraries(fcw_tests PRIVATE fcw GTest::gtest_main)
target_compile_definitions(fcw_tests PRIVATE FCW_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden"
                                           FCW_GOLDEN_LOCAL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden_local")
include(GoogleTest)
gtest_discover_tests(fcw_tests)

# Board backends (fcw/board.hpp). Capture + display are compile-checked on the PC when GStreamer and
# libdrm are installed; the XRT / VART accelerators build only on the board image (FCW_BOARD=ON).
find_package(PkgConfig QUIET)
if(FCW_BOARD_IO AND PkgConfig_FOUND)
  pkg_check_modules(GST IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0)
  pkg_check_modules(DRM IMPORTED_TARGET libdrm)
  if(GST_FOUND AND DRM_FOUND)
    add_library(fcw_board_io STATIC ${CMAKE_CURRENT_SOURCE_DIR}/backends/board/gst_capture.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/backends/board/drm_display.cpp)
    target_link_libraries(fcw_board_io PUBLIC fcw PkgConfig::GST PkgConfig::DRM)
    target_compile_options(fcw_board_io PRIVATE -Wall -Wextra)
    add_executable(fcw_board_io_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/board_io/test_gst_capture.cpp
                                      ${CMAKE_CURRENT_SOURCE_DIR}/tests/board_io/test_drm_display.cpp)
    target_link_libraries(fcw_board_io_tests PRIVATE fcw_board_io GTest::gtest_main)
    target_compile_definitions(fcw_board_io_tests PRIVATE FCW_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data")
  endif()
endif()
if(FCW_BOARD)
  find_package(XRT REQUIRED)
  find_package(vart REQUIRED COMPONENTS runner)
  find_package(xir REQUIRED)
  add_library(fcw_board STATIC ${CMAKE_CURRENT_SOURCE_DIR}/backends/board/xrt_accel.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/backends/board/vart_yolo.cpp)
  target_compile_definitions(fcw_board PUBLIC FCW_BOARD)
  target_link_libraries(fcw_board PUBLIC fcw_board_io XRT::xrt_coreutil vart::runner xir::xir)
  target_link_libraries(fcw_app PRIVATE fcw_board)
endif()
