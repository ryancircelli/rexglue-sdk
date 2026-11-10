# refxenos: standalone build (does not need the SDK's submodules).
#   cmake -S tools/gputest/refxenos -B out/refxenos -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   ~/Games/Xbox360/bo2-build/runq/bin/buildq -- ninja -C out/refxenos
cmake_minimum_required(VERSION 3.20)
project(refxenos CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Exact IEEE semantics: no fast-math, no FMA contraction (Xenos has separate
# multiply and add roundings unless an instruction says otherwise).
add_compile_options(-ffp-contract=off -fno-fast-math -Wall -Wno-unused-function)

get_filename_component(SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
file(GLOB_RECURSE REFXENOS_SOURCES CONFIGURE_DEPENDS
     "${SDK_ROOT}/src/graphics/refxenos/*.cpp" "${SDK_ROOT}/src/graphics/refxenos/*.cc")
# The texture library (texconf, branch refxenos-tex) ships its own self-test
# with a main(); build it as its own executable, not into the core.
set(REFXENOS_TEX_SELFTEST "${SDK_ROOT}/src/graphics/refxenos/texture/ref_texture_selftest.cc")
list(REMOVE_ITEM REFXENOS_SOURCES "${REFXENOS_TEX_SELFTEST}")
# SDK sources with no dependencies beyond headers (register names, ucode
# helpers, xenos float conversions).
set(SDK_GRAPHICS_SOURCES
  "${SDK_ROOT}/src/graphics/register_file.cpp"
  "${SDK_ROOT}/src/graphics/registers.cpp"
  "${SDK_ROOT}/src/graphics/xenos.cpp"
  "${SDK_ROOT}/src/graphics/format/ucode.cpp"
  "${SDK_ROOT}/src/core/math_gcc.cpp")
# OBJECT, not STATIC: rules register through static initializers, and a
# static archive would drop translation units nothing references, losing their
# rules from the registry (and from --rules-md / rules.json).
add_library(refxenos_core OBJECT ${REFXENOS_SOURCES} ${SDK_GRAPHICS_SOURCES})
target_include_directories(refxenos_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/shim"
  "${SDK_ROOT}/include"
  "${SDK_ROOT}/src/graphics/refxenos"
  "${SDK_ROOT}/src/graphics/refxenos/third_party")
find_package(ZLIB REQUIRED)
target_link_libraries(refxenos_core PUBLIC ZLIB::ZLIB)

file(GLOB REFXENOS_TOOL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tool/*.cpp")
if(REFXENOS_TOOL_SOURCES)
  add_executable(refxenos ${REFXENOS_TOOL_SOURCES})
  target_link_libraries(refxenos PRIVATE refxenos_core)
endif()

add_executable(refxenos_tex_selftest ${REFXENOS_TEX_SELFTEST})
target_link_libraries(refxenos_tex_selftest PRIVATE refxenos_core)

file(GLOB REFXENOS_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
foreach(t ${REFXENOS_TESTS})
  get_filename_component(n ${t} NAME_WE)
  add_executable(refxenos_${n} ${t})
  target_link_libraries(refxenos_${n} PRIVATE refxenos_core)
  if(NOT n STREQUAL "test_shaderjob")
    add_test(NAME ${n} COMMAND refxenos_${n})
  endif()
endforeach()
