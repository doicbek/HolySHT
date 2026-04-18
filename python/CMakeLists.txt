cmake_minimum_required(VERSION 3.15)
project(holysht_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

set(DUCC_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

# Minimal DUCC static library (same source set as MEX build).
set(DUCC_SOURCES
    ${DUCC_SRC_DIR}/ducc0/sht/sht.cc
    ${DUCC_SRC_DIR}/ducc0/infra/threading.cc
    ${DUCC_SRC_DIR}/ducc0/infra/mav.cc
    ${DUCC_SRC_DIR}/ducc0/infra/string_utils.cc
    ${DUCC_SRC_DIR}/ducc0/math/gl_integrator.cc
    ${DUCC_SRC_DIR}/ducc0/fft/fft_inst1.cc
    ${DUCC_SRC_DIR}/ducc0/fft/fft_inst2.cc
)

add_library(ducc0_static STATIC ${DUCC_SOURCES})
target_include_directories(ducc0_static PUBLIC ${DUCC_SRC_DIR})
target_compile_features(ducc0_static PUBLIC cxx_std_17)
set_target_properties(ducc0_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(ducc0_static PRIVATE -O3 -march=native -mno-avx512f -ffast-math)
endif()

# pybind11 extension module
pybind11_add_module(_holysht_core ${CMAKE_CURRENT_SOURCE_DIR}/holysht_pybind.cpp)
target_link_libraries(_holysht_core PRIVATE ducc0_static)
target_include_directories(_holysht_core PRIVATE ${DUCC_SRC_DIR})
# Suppress false-positive GIL assertion with Python 3.14 free-threaded builds
target_compile_definitions(_holysht_core PRIVATE PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF)

install(TARGETS _holysht_core LIBRARY DESTINATION holysht)
