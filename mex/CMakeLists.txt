# CMakeLists.txt for holysht MATLAB MEX interface (minimal DUCC build).
cmake_minimum_required(VERSION 3.15)

project(holysht_matlab_mex)

# Find MATLAB
find_program(MATLAB_EXECUTABLE matlab PATHS
    "C:/Program Files/MATLAB/*/bin"
    "C:/Program Files (x86)/MATLAB/*/bin"
    "/usr/local/MATLAB/*/bin"
    "/Applications/MATLAB_*.app/bin"
    "$ENV{MATLAB_ROOT}/bin"
    NO_DEFAULT_PATH
)

if(NOT MATLAB_EXECUTABLE)
    message(FATAL_ERROR "MATLAB not found. Please set MATLAB_ROOT or add MATLAB to PATH")
endif()

get_filename_component(MATLAB_ROOT ${MATLAB_EXECUTABLE} DIRECTORY)
get_filename_component(MATLAB_ROOT ${MATLAB_ROOT} DIRECTORY)

find_program(MEX_EXECUTABLE mex PATHS "${MATLAB_ROOT}/bin" NO_DEFAULT_PATH)
if(NOT MEX_EXECUTABLE)
    message(FATAL_ERROR "MEX compiler not found")
endif()

set(MATLAB_INCLUDE_DIR "${MATLAB_ROOT}/extern/include")

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(MEX_EXT "mexw64")
    else()
        set(MEX_EXT "mexw32")
    endif()
elseif(APPLE)
    set(MEX_EXT "mexmaci64")
else()
    set(MEX_EXT "mexa64")
endif()

set(DUCC_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

include_directories(
    ${MATLAB_INCLUDE_DIR}
    ${DUCC_SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Minimal DUCC source set required for ring-based SHT only.
# Minimal DUCC source set required for ring-based SHT only.
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
set_target_properties(ducc0_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
if(MSVC)
    target_compile_options(ducc0_static PRIVATE /O2)
else()
    target_compile_options(ducc0_static PRIVATE -O3 -march=native -ffast-math)
endif()

if(WIN32)
    set(DUCC_LIB_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ducc0_static${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
    set(DUCC_LIB_FILE "${CMAKE_CURRENT_BINARY_DIR}/libducc0_static.a")
endif()

set(MEX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/holysht_map2alm_mex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/holysht_alm2map_mex.cpp
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX)
    set(WRAPPER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/mex_wrapper.sh")
    file(WRITE ${WRAPPER_SCRIPT} "#!/bin/bash
# Pass CXXFLAGS directly to MEX command.
HAS_CXXFLAGS=false
for arg in \"\$@\"; do
    if [[ \"\$arg\" == CXXFLAGS=* ]]; then
        HAS_CXXFLAGS=true
        break
    fi
done
if [ \"\$HAS_CXXFLAGS\" = false ]; then
    exec \"${MEX_EXECUTABLE}\" CXXFLAGS=\"-std=c++17 -fPIC -O3 -march=native -ffast-math\" \"\$@\"
else
    exec \"${MEX_EXECUTABLE}\" \"\$@\"
fi
")
    file(CHMOD ${WRAPPER_SCRIPT} FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

foreach(mex_file ${MEX_SOURCES})
    get_filename_component(mex_name ${mex_file} NAME_WE)
    set(MEX_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${mex_name}.${MEX_EXT}")

    if(UNIX)
        add_custom_command(
            OUTPUT ${MEX_OUTPUT}
            COMMAND ${WRAPPER_SCRIPT}
                -I${MATLAB_INCLUDE_DIR}
                -I${DUCC_SRC_DIR}
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -cxx
                -outdir ${CMAKE_CURRENT_BINARY_DIR}
                ${mex_file}
                -L${CMAKE_CURRENT_BINARY_DIR}
                -lducc0_static
            DEPENDS ${mex_file} ${CMAKE_CURRENT_SOURCE_DIR}/ducc0_mex_utils.h ${WRAPPER_SCRIPT} ducc0_static
            COMMENT "Building MEX file ${mex_name}.${MEX_EXT}"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT ${MEX_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E env CXXFLAGS="-std=c++17 /O2"
                ${MEX_EXECUTABLE}
                -I${MATLAB_INCLUDE_DIR}
                -I${DUCC_SRC_DIR}
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -cxx
                -outdir ${CMAKE_CURRENT_BINARY_DIR}
                ${mex_file}
                ${DUCC_LIB_FILE}
            DEPENDS ${mex_file} ${CMAKE_CURRENT_SOURCE_DIR}/ducc0_mex_utils.h ducc0_static
            COMMENT "Building MEX file ${mex_name}.${MEX_EXT}"
            VERBATIM
        )
    endif()

    add_custom_target(${mex_name}_mex ALL DEPENDS ${MEX_OUTPUT})
endforeach()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/*.${MEX_EXT}
    DESTINATION lib/matlab/mex
    COMPONENT mex
)

message(STATUS "MATLAB root: ${MATLAB_ROOT}")
message(STATUS "MATLAB include: ${MATLAB_INCLUDE_DIR}")
message(STATUS "MEX compiler: ${MEX_EXECUTABLE}")
message(STATUS "MEX extension: ${MEX_EXT}")
message(STATUS "DUCC source: ${DUCC_SRC_DIR}")
