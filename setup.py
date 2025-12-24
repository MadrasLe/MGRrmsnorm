"""
MEGA GEMM - High Performance CUDA Kernels for LLMs
===================================================
Setup script for building CUDA extensions.
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='megagemm',
    version='0.1.0',
    author='Gabriel Yogi',
    author_email='gabriel@example.com',
    description='High-performance CUDA kernels for RMSNorm and SwiGLU - faster than PyTorch native',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MadrasLe/MGRrmsnorm',
    license='MIT',
    
    packages=find_packages(),
    
    ext_modules=[
        CUDAExtension(
            name='rmsnorm_cuda_ops',
            sources=[
                'pytorch_binding/binding.cpp',
                'src/rmsnorm_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                ],
            },
        ),
    ],
    
    cmdclass={
        'build_ext': BuildExtension
    },
    
    install_requires=[
        'torch>=2.0',
        'triton>=2.0',
    ],
    
    extras_require={
        'dev': ['pytest', 'pytest-benchmark'],
    },
    
    python_requires='>=3.8',
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
