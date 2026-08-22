cmake_minimum_required(VERSION 3.18)

project(MicroGemm LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MICROGEMM_BUILD_CLI "Build the microgemm CLI" ON)
option(MICROGEMM_BUILD_TEXT "Build the microgemm-text CLI" ON)

if(ANDROID)
    set(MICROGEMM_BUILD_CONVERT_DEFAULT OFF)
else()
    set(MICROGEMM_BUILD_CONVERT_DEFAULT ON)
endif()
option(MICROGEMM_BUILD_CONVERT "Build the host-side microgemm-convert CLI" ${MICROGEMM_BUILD_CONVERT_DEFAULT})

find_package(Threads)

add_library(microgemm_core STATIC
    src/microgemm_format.c
    src/microgemm_runtime.c
    src/microgemm_ops_cpu.c
    src/microgemm_decode_cpu.c
    src/microgemm_model_i8.c
)
target_include_directories(microgemm_core PUBLIC include)
set_target_properties(microgemm_core PROPERTIES OUTPUT_NAME microgemm)

if(MSVC)
    target_compile_options(microgemm_core PRIVATE /W4 /D_CRT_SECURE_NO_WARNINGS)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(microgemm_core PRIVATE -Wall -Wextra)
endif()

if(ANDROID)
    target_compile_definitions(microgemm_core PUBLIC MICROGEMM_ANDROID=1)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(microgemm_core PRIVATE -Wno-unknown-pragmas)
    endif()
else()
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(microgemm_core PUBLIC OpenMP::OpenMP_C)
    endif()

    if(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|X86")
        target_compile_options(microgemm_core PRIVATE /arch:AVX2)
        target_compile_definitions(microgemm_core PRIVATE MICROGEMM_FORCE_AVX2=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
        target_compile_options(microgemm_core PRIVATE -march=native)
    endif()
endif()

if(NOT MSVC)
    target_link_libraries(microgemm_core PUBLIC m)
endif()

if(MICROGEMM_BUILD_CLI)
    add_executable(microgemm_cli src/microgemm_cli.c)
    target_link_libraries(microgemm_cli PRIVATE microgemm_core)
    set_target_properties(microgemm_cli PROPERTIES OUTPUT_NAME microgemm)
endif()

if(MICROGEMM_BUILD_TEXT)
    add_executable(microgemm_text src/microgemm_text.cpp)
    target_link_libraries(microgemm_text PRIVATE microgemm_core)
    if(Threads_FOUND)
        target_link_libraries(microgemm_text PRIVATE Threads::Threads)
    endif()
    set_target_properties(microgemm_text PROPERTIES OUTPUT_NAME microgemm-text)
endif()

if(MICROGEMM_BUILD_CONVERT)
    add_executable(microgemm_convert src/microgemm_convert.cpp)
    target_link_libraries(microgemm_convert PRIVATE microgemm_core)
    set_target_properties(microgemm_convert PROPERTIES OUTPUT_NAME microgemm-convert)
endif()
