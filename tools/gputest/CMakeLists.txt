# GPU test harness tools. Enabled with -DREXGLUE_BUILD_GPUTEST=ON.
# See tools/gputest/README.md.

set(REXGPU_SRC ${REXGLUE_ROOT}/src/graphics)

# rexgpu-shadertool: offline SPIR-V translation + reflection + CPU interpreter.
# The GPU plugin is a hidden-visibility shared library, so the translator
# sources are compiled into the tool directly.
if(REXGLUE_USE_VULKAN)
    add_executable(rexgpu-shadertool
        shadertool/shadertool.cpp
        ${REXGPU_SRC}/flags.cpp
        ${REXGPU_SRC}/register_file.cpp
        ${REXGPU_SRC}/registers.cpp
        ${REXGPU_SRC}/format/ucode.cpp
        ${REXGPU_SRC}/pipeline/shader/shader.cpp
        ${REXGPU_SRC}/pipeline/shader/translator.cpp
        ${REXGPU_SRC}/pipeline/shader/translator_disasm.cpp
        ${REXGPU_SRC}/pipeline/shader/interpreter.cpp
        ${REXGPU_SRC}/pipeline/shader/spirv_builder.cpp
        ${REXGPU_SRC}/pipeline/shader/spirv.cpp
        ${REXGPU_SRC}/pipeline/shader/spirv_translator.cpp
        ${REXGPU_SRC}/pipeline/shader/spirv_translator_alu.cpp
        ${REXGPU_SRC}/pipeline/shader/spirv_translator_fetch.cpp
        ${REXGPU_SRC}/pipeline/shader/spirv_translator_memexport.cpp
        ${REXGPU_SRC}/pipeline/shader/spirv_translator_rb.cpp
        ${REXGPU_SRC}/util/draw.cpp
    )
    target_include_directories(rexgpu-shadertool PRIVATE ${REXGLUE_ROOT})
    target_compile_options(rexgpu-shadertool PRIVATE -Wno-unused-variable)
    target_link_libraries(rexgpu-shadertool PRIVATE
        rexruntime
        xxHash::xxhash
        Vulkan::Headers
        glslang::SPIRV
        glslang
        MachineIndependent
        GenericCodeGen
        OSDependent
        OGLCompiler
    )
endif()

if(APPLE)
    # rexgpu-metal-runner: generic Metal job runner used by the differential
    # shader tests (renders a pixel shader over a cell grid, or runs a vertex
    # shader wrapped as a compute kernel, and dumps raw float outputs).
    add_executable(rexgpu-metal-runner shadertool/metal_runner.mm)
    target_compile_options(rexgpu-metal-runner PRIVATE -fobjc-arc)
    target_link_libraries(rexgpu-metal-runner PRIVATE "-framework Metal" "-framework Foundation")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/replay/CMakeLists.txt)
    add_subdirectory(replay)
endif()
