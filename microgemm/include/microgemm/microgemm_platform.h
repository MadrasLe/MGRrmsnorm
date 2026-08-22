#ifndef MICROGEMM_PLATFORM_H
#define MICROGEMM_PLATFORM_H

#if defined(_MSC_VER)
#define MICROGEMM_COMPILER_MSVC 1
#else
#define MICROGEMM_COMPILER_MSVC 0
#endif

#if defined(__ANDROID__)
#define MICROGEMM_PLATFORM_ANDROID 1
#else
#define MICROGEMM_PLATFORM_ANDROID 0
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MICROGEMM_ARCH_X86 1
#else
#define MICROGEMM_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define MICROGEMM_ARCH_ARM64 1
#else
#define MICROGEMM_ARCH_ARM64 0
#endif

#if MICROGEMM_ARCH_X86 && (defined(__AVX2__) || defined(_M_AVX2) || defined(MICROGEMM_FORCE_AVX2))
#define MICROGEMM_CPU_X86_AVX2 1
#else
#define MICROGEMM_CPU_X86_AVX2 0
#endif

#if MICROGEMM_ARCH_X86 && (defined(__FMA__) || defined(MICROGEMM_FORCE_FMA) || (MICROGEMM_COMPILER_MSVC && MICROGEMM_CPU_X86_AVX2))
#define MICROGEMM_CPU_X86_FMA 1
#else
#define MICROGEMM_CPU_X86_FMA 0
#endif

#if MICROGEMM_ARCH_X86 && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#define MICROGEMM_CPU_X86_AVX512_VNNI 1
#else
#define MICROGEMM_CPU_X86_AVX512_VNNI 0
#endif

#if MICROGEMM_ARCH_X86 && defined(__AVXVNNI__)
#define MICROGEMM_CPU_X86_AVX_VNNI 1
#else
#define MICROGEMM_CPU_X86_AVX_VNNI 0
#endif

#if MICROGEMM_ARCH_ARM64 && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64))
#define MICROGEMM_CPU_ARM64_NEON 1
#else
#define MICROGEMM_CPU_ARM64_NEON 0
#endif

#if defined(__ARM_FEATURE_DOTPROD)
#define MICROGEMM_CPU_ARM64_DOTPROD 1
#else
#define MICROGEMM_CPU_ARM64_DOTPROD 0
#endif

#if MICROGEMM_CPU_X86_AVX2 || MICROGEMM_CPU_X86_AVX512_VNNI || MICROGEMM_CPU_X86_AVX_VNNI
#include <immintrin.h>
#endif

#if MICROGEMM_CPU_ARM64_NEON
#include <arm_neon.h>
#endif

#endif
