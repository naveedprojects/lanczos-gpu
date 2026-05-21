"""Build script for gpu_eigsh CUDA+pybind11 extension."""

import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')


class CUDAExtension(Extension):
    """Extension that mixes .cu and .cpp files."""
    pass


class BuildExt(build_ext):
    """Custom build: compile .cu with nvcc, .cpp with g++, link together."""

    def build_extensions(self):
        for ext in self.extensions:
            self._build_cuda_ext(ext)

    def _build_cuda_ext(self, ext):
        import pybind11

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        # Separate .cu and .cpp sources
        cu_sources = [s for s in ext.sources if s.endswith('.cu')]
        cpp_sources = [s for s in ext.sources if s.endswith('.cpp')]

        # Compile .cu files with nvcc -> .o
        cu_objects = []
        for src in cu_sources:
            obj = src + '.o'
            cmd = [
                f'{CUDA_HOME}/bin/nvcc',
                '-O3', '-arch=sm_86', '-std=c++17',
                '--compiler-options', '-fPIC',
                '-I', SRC_DIR,
                '-I', pybind11.get_include(),
                '-I', f'{CUDA_HOME}/include',
                '-c', src, '-o', obj,
            ]
            print(' '.join(cmd))
            subprocess.check_call(cmd)
            cu_objects.append(obj)

        # Compile .cpp files with g++ -> .o
        cpp_objects = []
        for src in cpp_sources:
            obj = src + '.o'
            cmd = [
                'g++', '-O3', '-std=c++17', '-fPIC', '-shared',
                '-I', SRC_DIR,
                '-I', pybind11.get_include(),
                '-I', pybind11.get_include(user=True),
                f'-I{sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor}',
                '-I', f'{CUDA_HOME}/include',
                '-c', src, '-o', obj,
            ]
            print(' '.join(cmd))
            subprocess.check_call(cmd)
            cpp_objects.append(obj)

        # Compile CUDA source files from src/
        src_cu_files = [
            os.path.join(SRC_DIR, 'irlm_lanczos.cu'),
            os.path.join(SRC_DIR, 'cast_kernels.cu'),
        ]
        src_objects = []
        build_tmp = self.build_temp or '.'
        os.makedirs(build_tmp, exist_ok=True)
        for src in src_cu_files:
            basename = os.path.splitext(os.path.basename(src))[0]
            obj = os.path.join(build_tmp, f'{basename}.o')
            cmd = [
                f'{CUDA_HOME}/bin/nvcc',
                '-O3', '-arch=sm_86', '-std=c++17',
                '--compiler-options', '-fPIC',
                '-I', SRC_DIR,
                '-c', src, '-o', obj,
            ]
            print(' '.join(cmd))
            subprocess.check_call(cmd)
            src_objects.append(obj)

        # Link everything into shared library
        all_objects = cu_objects + cpp_objects + src_objects
        link_cmd = [
            'g++', '-shared', '-o', ext_path,
        ] + all_objects + [
            f'-L{CUDA_HOME}/lib64',
            '-lcudart', '-lcusparse', '-lcublas', '-lcusolver',
            '-llapack', '-lblas',
            '-ldl',
        ]
        print(' '.join(link_cmd))
        subprocess.check_call(link_cmd)


setup(
    name='gpu_eigsh',
    version='0.2.0',
    description='GPU-accelerated differentiable sparse eigenvalue solver '
                '(ARPACK-quality IRLM on CUDA with implicit differentiation)',
    packages=['gpu_eigsh'],
    ext_modules=[
        CUDAExtension(
            'gpu_eigsh._core',
            sources=[
                'gpu_eigsh/_cuda_eigsh.cu',
                'gpu_eigsh/_cuda_funm.cu',
                'gpu_eigsh/_bindings.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExt},
    install_requires=['numpy', 'scipy'],
    extras_require={
        'torch': ['torch'],
    },
    python_requires='>=3.8',
)
