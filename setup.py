# Copyright (c) 2023, Tri Dao.

import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from typing import Optional, Union

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "--offload-arch=xcore1000" to files ending in '_xcore1000.cu',
# "--offload-arch=xcore1500" to files ending in '_xcore1500.cu'
from torch.utils.cpp_extension import (
    IS_HIP_EXTENSION,
    COMMON_HIP_FLAGS,
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    get_cxx_compiler,
    _join_rocm_home,
    _join_cuda_home,
    _is_cuda_file,
    _maybe_write,
)
def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda,
                      **kwargs,  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension
                      ) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if "PYTORCH_NVCC" in os.environ:
            nvcc = os.getenv("PYTORCH_NVCC")    # user can set nvcc compiler with ccache using the environment variable here
        else:
            if IS_HIP_EXTENSION:
                nvcc = _join_rocm_home('bin', 'hipcc')
            else:
                nvcc = _join_cuda_home('bin', 'nvcc')
        config.append(f'nvcc = {nvcc}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        maca_post_cflags_xcore1000 = cuda_post_cflags + ['--offload-arch=xcore1000']
        flags.append(f'maca_post_cflags_xcore1000 = {" ".join(maca_post_cflags_xcore1000)}')
        maca_post_cflags_xcore1500 = cuda_post_cflags + ['--offload-arch=xcore1500']
        flags.append(f'maca_post_cflags_xcore1500 = {" ".join(maca_post_cflags_xcore1500)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        maca_compile_rule_xcore1000 = ['rule maca_compile_xcore1000'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $maca_post_cflags_xcore1000'
        ]
        maca_compile_rule_xcore1500 = ['rule maca_compile_xcore1500'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $maca_post_cflags_xcore1500'
        ]
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        if is_cuda_source:
            if source_file.endswith('_xcore1000.cu'):
                rule = 'maca_compile_xcore1000'
            elif source_file.endswith('_xcore1500.cu'):
                rule = 'maca_compile_xcore1500'
            else:
                rule = 'cuda_compile'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(maca_compile_rule_xcore1000)  # type: ignore[possibly-undefined]
        blocks.append(maca_compile_rule_xcore1500)  # type: ignore[possibly-undefined]
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)

# Monkey patching
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flash_mla"

BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"

cmdclass = {}
ext_modules = []

flash_mla_root = Path(__file__).parent.parent

def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    # raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    # output = raw_output.split()
    # release_idx = output.index("release") + 1
    # bare_metal_version = parse(output[release_idx].split(",")[0])
    raw_output = "nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2023 NVIDIA Corporation Built on Mon_Apr__3_17:16:06_PDT_2023 Cuda compilation tools, release 12.1, V12.1.105 Build cuda_12.1.r12.1/compiler.32688072_0"
    bare_metal_version = Version("12.1")
    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

def get_arch_list(build_projects: str):
    projects_list = build_projects.split(':')
    all_arch_list = ["xcore1000", "xcore1500"]
    arch_list = []
    for project in projects_list:
        if project == "C500":
            arch_list.append("xcore1000")
        elif project == "C600":
            arch_list.append("xcore1500")
    arch_list = list(set(arch_list))
    if len(arch_list) == 0:
        arch_list = all_arch_list
    assert 0 < len(arch_list) and len(arch_list) <= len(all_arch_list)
    return arch_list



# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source
if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                "FlashAttention is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    lib_dir = Path(CUDA_HOME).parent.parent / "lib"
    libraries=["mcblas"]
    extra_objects = ['{}/lib{}.so'.format(lib_dir, l) for l in libraries]
    # extra_objects.extend([f for f in obj_lists if f.endswith('.o')])

    feature_flag = []
    src_file = [
        "csrc/flash_api/flash_api.cpp",
        "csrc/flash_run/run_mla_fwd.cpp",
        "csrc/instantiations/flash_fwd_mla_metadata.cu",
        "csrc/utils/print_parameter.cpp",
        "csrc/utils/process_str.cpp",
        "csrc/utils/logger.cpp",
    ]
    xcore1000_src_file = [
        # dense
        "csrc/instantiations/xcore1000/flash_fwd_mla_hdimqk576_hdimv512_m16n16_bf16_True_True_1stage_xcore1000.cu",
        "csrc/instantiations/xcore1000/flash_fwd_mla_hdimqk576_hdimv512_m16n16_fp16_True_True_1stage_xcore1000.cu",
        "csrc/instantiations/xcore1000/flash_fwd_mla_hdimqk576_hdimv512_m16n16_bf16_True_True_split_xcore1000.cu",
        "csrc/instantiations/xcore1000/flash_fwd_mla_hdimqk576_hdimv512_m16n16_fp16_True_True_split_xcore1000.cu",
        "csrc/instantiations/xcore1000/flash_fwd_mla_hdimqk576_hdimv512_m32n16_bf16_True_True_split_xcore1000.cu",
        "csrc/instantiations/xcore1000/flash_fwd_mla_hdimqk576_hdimv512_m32n16_fp16_True_True_split_xcore1000.cu",
        "csrc/instantiations/xcore1000/flash_fwd_mla_hdimqk576_hdimv512_m64n16_bf16_True_True_split_xcore1000.cu",
        "csrc/instantiations/xcore1000/flash_fwd_mla_hdimqk576_hdimv512_m64n16_fp16_True_True_split_xcore1000.cu",
        # sparse
        "csrc/instantiations/xcore1000/flash_fwd_prefill_hdimqk576_hdimv512_m64n16_bf16_True_True_split_xcore1000.cu",
        # "csrc/instantiations/xcore1000/flash_fwd_prefill_hdimqk576_hdimv512_m64n16_fp16_True_True_split_xcore1000.cu", # sparse prefill only supports bf16
        "csrc/instantiations/xcore1000/flash_fwd_sparse_mla_hdimqk576_hdimv512_m64n16_bf16_True_True_split_xcore1000.cu",
        "csrc/instantiations/xcore1000/flash_fwd_sparse_mla_hdimqk576_hdimv512_m64n16_fp16_True_True_split_xcore1000.cu",
    ]
    xcore1500_src_file = [
        # dense
        # xcore1500 recommand config is blockN 32
        # "csrc/instantiations/xcore1500/flash_fwd_mla_hdimqk576_hdimv512_m64n16_bf16_True_True_split_xcore1500.cu",
        # "csrc/instantiations/xcore1500/flash_fwd_mla_hdimqk576_hdimv512_m64n16_fp16_True_True_split_xcore1500.cu",
        "csrc/instantiations/xcore1500/flash_fwd_mla_hdimqk576_hdimv512_m64n32_bf16_True_True_split_xcore1500.cu",
        "csrc/instantiations/xcore1500/flash_fwd_mla_hdimqk576_hdimv512_m64n32_fp16_True_True_split_xcore1500.cu",
        # sparse
        "csrc/instantiations/xcore1500/flash_fwd_prefill_hdimqk576_hdimv512_m64n16_bf16_True_True_split_xcore1500.cu",
        "csrc/instantiations/xcore1500/flash_fwd_sparse_mla_hdimqk576_hdimv512_m64n16_bf16_True_True_split_xcore1500.cu",
        "csrc/instantiations/xcore1500/flash_fwd_sparse_mla_hdimqk576_hdimv512_m64n16_fp16_True_True_split_xcore1500.cu",
    ]

    build_projects = os.getenv("FLASHATTN_BUILD_PROJECTS", "C500:C600")
    arch_list = get_arch_list(build_projects)
    print(f"build arch list is {arch_list}")
    only_one_arch = len(arch_list) == 1
    # we support build all arch or build only one specific arch
    cucc_targets = ""
    for arch in arch_list:
        if cucc_targets == "":
            cucc_targets = arch
        else:
            cucc_targets = cucc_targets + "," + arch
        if arch == "xcore1000":
            if only_one_arch:
                feature_flag.append("-DXCORE1000")
            src_file.extend(xcore1000_src_file)
        elif arch == "xcore1500":
            if only_one_arch:
                feature_flag.append("-DXCORE1500")
            src_file.extend(xcore1500_src_file)
    # set CUCC_TARGETS for common kernel (e.g. flash_fwd_mla_metadata.cu)
    os.environ['CUCC_TARGETS'] = cucc_targets
    print("CUCC_TARGETS=",os.getenv('CUCC_TARGETS'))

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    ext_modules.append(
        CUDAExtension(
            name="flash_mla_cuda",
            sources=src_file,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-w"] + generator_flag + feature_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-w",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "-fno-strict-aliasing",
                        "-mllvm",
                        "-enable-tbaa=false",
                        "-mllvm",
                        "-metaxgpu-disable-bsm-offset=0",#try to merge ldg/lds
                        "-mllvm",
                        "-metaxgpu-sink-gep=true",
                        "-D__FAST_HALF_CVT__",
                        "-Xclang",
                        "-menable-no-nans",
                        "-mllvm",
                        "--metaxgpu-prefer-scsel=true",
                        "-mllvm",
                        "-metaxgpu-mmasched=true",
                        "-mllvm",
                        "-metaxgpu-internalize-symbols",
                    ]
                    + generator_flag
                    + cc_flag
                    + feature_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_api",
                Path(this_dir) / "csrc" / "flash_dispatch",
                Path(this_dir) / "csrc" / "flash_kernel" ,
                Path(this_dir) / "csrc" / "flash_kernel" / "feature",
                Path(this_dir) / "csrc" / "flash_run",
                Path(this_dir) / "csrc" / "utils",
                Path(this_dir) / "csrc" / "mctlass" / "include",
            ],
            extra_objects = extra_objects,
        )
    )

def get_package_version():
    with open(Path(this_dir) / "flash_mla" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = str(ast.literal_eval(version_match.group(1)))
    maca_version = os.environ.get("CORE_MODULE_MACA_VERSION")
    if maca_version:
        return public_version + "+metax" + maca_version
    else:
        return public_version + "+"

def get_torch_version():
    torch_version_raw = parse(torch.__version__)
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    return torch_version

def get_wheel_url():
    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
    torch_cuda_version = parse(torch.version.cuda)
    # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.2
    # to save CI time. Minor versions should be compatible.
    torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.2")
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    flash_version = get_package_version()
    # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    torch_version = get_torch_version()
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f"{PACKAGE_NAME}-{flash_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{flash_version}", wheel_name=wheel_filename)
    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        #try:
        #    urllib.request.urlretrieve(wheel_url, wheel_filename)

        #     # Make the archive
        #     # Lifted from the root wheel processing command
        #     # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
        #     if not os.path.exists(self.dist_dir):
        #         os.makedirs(self.dist_dir)

        #     impl_tag, abi_tag, plat_tag = self.get_tag()
        #     archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

        #     wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
        #     print("Raw wheel path", wheel_path)
        #     os.rename(wheel_filename, wheel_path)
        # except urllib.error.HTTPError:
        #     print("Precompiled wheel not found. Building from source...")
        #     # If the wheel could not be downloaded, build from source
        #     super().run()
        print("Building wheel from source...")
        super().run()


setup(
    name=PACKAGE_NAME,
    version=get_package_version() + "torch" + get_torch_version() if get_torch_version() else get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "tools",
            "dist",
            "docs",
            "benchmarks",
            "flash_mla.egg-info",
        )
    ),
    author="Tri Dao, DeepSeek",
    author_email="trid@cs.stanford.edu",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention & flashMLA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
