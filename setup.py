from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rmsnorm_cuda',
    ext_modules=[
        CUDAExtension('rmsnorm_cuda_ops', [
            'pytorch_binding/binding.cpp',
            'src/rmsnorm_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
