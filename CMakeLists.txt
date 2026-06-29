# CMakeLists.txt  gzstd (hybrid CPU+GPU Zstd) with robust Zstd fallback & RPATH hardening

cmake_minimum_required(VERSION 3.18)
project(gzstd LANGUAGES CXX)

# ---------------- Options ----------------
option(USE_NVCOMP "Enable nvCOMP (CUDA) GPU backend if found" ON)
option(BUILD_STATIC "Link libraries statically for portable binaries" OFF)

# ---------------- Language / warnings ----------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/W3)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations)
endif()

# Threads (for std::thread and -pthread on Unix)
find_package(Threads REQUIRED)

# =====================================================
#                      Zstd
# Try, in order:
#   1) CMake's FindZSTD (target ZSTD::ZSTD)
#   2) pkg-config 'zstd'
#   3) Manual fallback that looks in common prefixes (e.g., $CONDA_PREFIX)
#      and allows -DZSTD_INCLUDE_DIR / -DZSTD_LIBRARY overrides.
#
# When BUILD_STATIC is ON, prefer libzstd.a over libzstd.so.
# =====================================================

# Helper: preferred library name list based on BUILD_STATIC
if(BUILD_STATIC)
  set(_ZSTD_LIB_NAMES zstd_static zstd)      # prefer static archive
  set(_ZSTD_LIB_SUFFIXES lib lib64)
else()
  set(_ZSTD_LIB_NAMES zstd)
  set(_ZSTD_LIB_SUFFIXES lib lib64)
endif()

# 1) Built-in finder (provides target ZSTD::ZSTD)
find_package(ZSTD QUIET)

# 2) pkg-config fallback (synthesizes target when found)
if(NOT ZSTD_FOUND)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(ZSTD_PKG zstd IMPORTED_TARGET)
    if(ZSTD_PKG_FOUND)
      add_library(ZSTD::ZSTD INTERFACE IMPORTED)
      target_include_directories(ZSTD::ZSTD INTERFACE ${ZSTD_PKG_INCLUDE_DIRS})
      if(BUILD_STATIC)
        target_link_libraries(ZSTD::ZSTD INTERFACE ${ZSTD_PKG_STATIC_LIBRARIES})
      else()
        target_link_libraries(ZSTD::ZSTD INTERFACE ${ZSTD_PKG_LINK_LIBRARIES})
      endif()
      message(STATUS "Zstd found via pkg-config: ${ZSTD_PKG_LINK_LIBRARIES}")
    endif()
  endif()
endif()

# 3) Manual fallback (works even without zstd.pc)
# Hints: user-supplied variables, then environment prefixes.
if(NOT TARGET ZSTD::ZSTD)
  # Allow explicit overrides:
  #   -DZSTD_INCLUDE_DIR=/path/include  -DZSTD_LIBRARY=/path/lib/libzstd.so
  if(NOT ZSTD_INCLUDE_DIR)
    find_path(ZSTD_INCLUDE_DIR zstd.h
              HINTS ${ZSTD_ROOT} ${ZSTD_PREFIX}
                    ENV CONDA_PREFIX ENV ZSTD_ROOT ENV ZSTD_PREFIX
              PATH_SUFFIXES include)
  endif()

  if(NOT ZSTD_LIBRARY)
    # When BUILD_STATIC is ON, force find_library to prefer .a over .so so
    # we pick up libzstd.a (which Ubuntu's libzstd-dev does ship) instead of
    # libzstd.so.  Without this, .so wins and the binary ends up dynamically
    # linked to libzstd.
    if(BUILD_STATIC)
      set(_zstd_orig_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
      set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    endif()
    find_library(ZSTD_LIBRARY NAMES ${_ZSTD_LIB_NAMES}
                 HINTS ${ZSTD_ROOT} ${ZSTD_PREFIX}
                       ENV CONDA_PREFIX ENV ZSTD_ROOT ENV ZSTD_PREFIX
                 PATH_SUFFIXES ${_ZSTD_LIB_SUFFIXES})
    if(BUILD_STATIC)
      set(CMAKE_FIND_LIBRARY_SUFFIXES "${_zstd_orig_suffixes}")
    endif()
  endif()

  if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
    add_library(ZSTD::ZSTD INTERFACE IMPORTED)
    target_include_directories(ZSTD::ZSTD INTERFACE ${ZSTD_INCLUDE_DIR})
    target_link_libraries(ZSTD::ZSTD INTERFACE ${ZSTD_LIBRARY})
    message(STATUS "Zstd found (manual): ${ZSTD_LIBRARY}")
  endif()
endif()

# Hard fail if we still don't have Zstd
if(NOT TARGET ZSTD::ZSTD)
  message(FATAL_ERROR
    "Zstd not found. Install headers+libs (e.g., conda-forge:zstd or libzstd-dev) "
    "or pass -DZSTD_INCLUDE_DIR & -DZSTD_LIBRARY.")
endif()

# =====================================================
#                nvCOMP + CUDA (optional)
# When BUILD_STATIC is ON, prefer static nvCOMP and
# static CUDA runtime (cudart_static).
# =====================================================
set(HAVE_NVCOMP FALSE)
if(USE_NVCOMP)
  # CUDA runtime
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    # Hints: NVCOMP_ROOT or CONDA_PREFIX
    # NVCOMP_ROOT can be passed as either a CMake cache variable
    # (-DNVCOMP_ROOT=/path) or an environment variable.  CONDA_PREFIX is
    # checked from the environment as a fallback for conda-installed nvCOMP.
    find_path(NVCOMP_INCLUDE_DIR
      NAMES nvcomp/zstd.h
      HINTS ${NVCOMP_ROOT} $ENV{NVCOMP_ROOT} $ENV{CONDA_PREFIX}
      PATH_SUFFIXES include)

    if(BUILD_STATIC)
      # Prefer libnvcomp_static.a; fall back to shared if user only has
      # conda-forge's package (which is shared-only).  NVIDIA's official
      # tarball at https://developer.download.nvidia.com/compute/nvcomp/redist/
      # ships libnvcomp_static.a.  See BUILD.md for the recipe.
      find_library(NVCOMP_LIBRARY
        NAMES nvcomp_static nvcomp
        HINTS ${NVCOMP_ROOT} $ENV{NVCOMP_ROOT} $ENV{CONDA_PREFIX}
        PATH_SUFFIXES lib lib64)
    else()
      find_library(NVCOMP_LIBRARY
        NAMES nvcomp
        HINTS ${NVCOMP_ROOT} $ENV{NVCOMP_ROOT} $ENV{CONDA_PREFIX}
        PATH_SUFFIXES lib lib64)
    endif()

    if(NVCOMP_INCLUDE_DIR AND NVCOMP_LIBRARY)
      set(HAVE_NVCOMP TRUE)
      message(STATUS "nvCOMP found: ${NVCOMP_LIBRARY}")
      # Warn when BUILD_STATIC requested but only shared nvCOMP was found
      if(BUILD_STATIC AND NVCOMP_LIBRARY MATCHES "\\.(so|dylib)")
        set(NVCOMP_IS_SHARED TRUE)
        message(WARNING
          "BUILD_STATIC is ON but only shared nvCOMP was found (${NVCOMP_LIBRARY}). "
          "The binary will still depend on libnvcomp.so at runtime. "
          "Run 'make install' or 'cmake --install' to bundle the .so next to the binary.")
      endif()
    else()
      message(STATUS "nvCOMP not found; building CPU-only (set NVCOMP_ROOT or CONDA_PREFIX if available)")
    endif()
  else()
    message(STATUS "CUDAToolkit not found; building CPU-only")
  endif()
endif()

# Enable the CUDA compiler only for the GPU build — the one custom kernel
# (gpuverify.cu, the --verify byte compare) is compiled by nvcc.  The CPU-only
# build stays LANGUAGES CXX (no nvcc required).
if(HAVE_NVCOMP)
  # Record whether the user pinned architectures explicitly
  # (-DCMAKE_CUDA_ARCHITECTURES=...).  This MUST be checked BEFORE
  # enable_language(CUDA): that call seeds a cached compiler default, after which
  # the variable always looks "defined" and we could no longer tell a user
  # override from CMake's fallback.  The architecture list itself is applied to
  # the gzstd target further down — once enable_language has told us the nvcc
  # version, which gates the newest -real arches.
  if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(GZSTD_CUDA_ARCH_USER_SET TRUE)
  else()
    set(GZSTD_CUDA_ARCH_USER_SET FALSE)
  endif()
  enable_language(CUDA)
endif()

# =====================================================
#          NVML (GPU utilization monitoring)
# NVML is part of the NVIDIA driver and CANNOT be statically linked.
# Linking even the toolkit's stub writes DT_NEEDED libnvidia-ml.so.1
# into the binary, and the ELF loader requires that library AT STARTUP
# — the portable binary refused to start on driver-less machines.
# Since v0.13.55 gzstd dlopen()s NVML at runtime (loader shim in
# gzstd.cpp): no link-time dependency, graceful fallback to free-VRAM
# ranking / sysfs PCIe probing when the driver is absent.  All it
# needs at build time is <dlfcn.h>.
# =====================================================
set(HAVE_NVML FALSE)
if(HAVE_NVCOMP AND CUDAToolkit_FOUND AND UNIX)
  set(HAVE_NVML TRUE)
  message(STATUS "NVML: runtime dlopen loader enabled (no link-time dependency)")
endif()

# =====================================================
#                     Target
# =====================================================
add_executable(gzstd gzstd.cpp)
target_compile_features(gzstd PRIVATE cxx_std_17)

# Include/defines
if(HAVE_NVCOMP)
  target_compile_definitions(gzstd PRIVATE HAVE_NVCOMP)
  target_include_directories(gzstd PRIVATE ${NVCOMP_INCLUDE_DIR})
  # The GPU-verify byte-compare kernel (nvcc-compiled).
  target_sources(gzstd PRIVATE gpuverify.cu)

  # Goal: the GPU-verify kernel must run on every GPU nvCOMP itself supports,
  # not just the boxes we test on.  Embed native SASS (-real) for the cards
  # people actually run (Turing through Blackwell) AND a low virtual-arch PTX
  # (75-virtual).  PTX is forward-compatible: a compute_75 image JITs onto ANY
  # device with capability >= 7.5 — the in-between parts we don't list
  # (Ampere sm_87, …) and every future architecture beyond Blackwell.  So the
  # virtual entry future-proofs the kernel without re-listing each new arch; the
  # real cubins are just a JIT-skip optimization for known cards.  Applied as a
  # target property (not the CMAKE_ variable) so it overrides the default CMake
  # seeds at enable_language regardless.  Skipped when the user pinned arches.
  if(NOT GZSTD_CUDA_ARCH_USER_SET)
    set(_gz_arches 75-real 80-real 86-real 89-real 90-real)
    # sm_100 (Blackwell datacenter) and sm_120 (Blackwell consumer) can only be
    # emitted as native cubins by nvcc >= 12.8.  On older toolkits we skip them
    # rather than failing the build — the compute_75 PTX below already JITs onto
    # those parts, so coverage is unchanged, only the one-time launch JIT remains.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
      list(APPEND _gz_arches 100-real 120-real)
    endif()
    list(APPEND _gz_arches 75-virtual)
    set_target_properties(gzstd PROPERTIES CUDA_ARCHITECTURES "${_gz_arches}")
    message(STATUS "gzstd verify-kernel CUDA architectures: ${_gz_arches}")
  endif()
endif()
if(HAVE_NVML)
  target_compile_definitions(gzstd PRIVATE HAVE_NVML)
endif()

# POSIX ACL support for --acls (libacl).  The --xattrs syscalls live in glibc
# and need no library; only ACLs require libacl.  Optional: without it, gzstd
# builds fine and --acls/--xattrs error at runtime.  When BUILD_STATIC, prefer
# the .a archives and also link libattr (libacl's dependency).
if(UNIX AND NOT APPLE)
  if(BUILD_STATIC)
    set(_SAVED_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
  find_path(ACL_INCLUDE_DIR sys/acl.h)
  find_library(ACL_LIBRARY NAMES acl)
  find_library(ATTR_LIBRARY NAMES attr)
  if(BUILD_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_SAVED_SUFFIXES})
  endif()
  if(ACL_INCLUDE_DIR AND ACL_LIBRARY)
    target_compile_definitions(gzstd PRIVATE GZSTD_HAVE_ACL=1)
    target_include_directories(gzstd PRIVATE ${ACL_INCLUDE_DIR})
    target_link_libraries(gzstd PRIVATE ${ACL_LIBRARY})
    if(BUILD_STATIC AND ATTR_LIBRARY)
      target_link_libraries(gzstd PRIVATE ${ATTR_LIBRARY})
    endif()
    message(STATUS "ACL/xattr support: enabled (${ACL_LIBRARY})")
  else()
    message(STATUS "ACL/xattr support: disabled (libacl not found; --acls/--xattrs unavailable)")
  endif()
endif()

# =====================================================
#                     Linkage
# =====================================================
if(HAVE_NVCOMP)
  # CUDA runtime: static when BUILD_STATIC, shared otherwise
  if(BUILD_STATIC)
    set(_CUDA_RT CUDA::cudart_static)
  else()
    set(_CUDA_RT CUDA::cudart)
  endif()

  # Core link libraries
  target_link_libraries(gzstd PRIVATE
    ZSTD::ZSTD
    Threads::Threads
    ${_CUDA_RT}
    ${NVCOMP_LIBRARY})

  # dl: the NVML runtime loader (dlopen) and static cudart both need it
  if(UNIX)
    target_link_libraries(gzstd PRIVATE dl)
  endif()
  # Static CUDA runtime additionally needs rt on Linux
  if(BUILD_STATIC AND UNIX)
    target_link_libraries(gzstd PRIVATE rt)
  endif()
else()
  target_link_libraries(gzstd PRIVATE
    ZSTD::ZSTD
    Threads::Threads)
endif()

# Static linking: prefer static libstdc++ and libgcc for maximum portability.
# The resulting binary only needs glibc (2.17+) at runtime; the NVIDIA driver
# is optional (CUDA and NVML are discovered at runtime, with CPU fallback).
if(BUILD_STATIC AND NOT MSVC)
  target_link_options(gzstd PRIVATE
    -static-libgcc
    -static-libstdc++)
endif()

# =====================================================
#                     RPATH
# Make ./build/gzstd run cleanly with Conda/system libs.
# When BUILD_STATIC is ON, RPATH is less important since
# most dependencies are linked statically (NVML is dlopen'd
# at runtime, so it needs no RPATH at all).
# =====================================================
if(UNIX AND NOT APPLE)
  # Base RPATH entries near the binary
  set(_RPATH_BASE "$ORIGIN;$ORIGIN/..;$ORIGIN/../lib;$ORIGIN/../lib64;$ORIGIN/lib")

  # Conda awareness (optional)
  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _RPATH_BASE "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/lib64")
  endif()

  set_target_properties(gzstd PROPERTIES
    BUILD_RPATH "${_RPATH_BASE}"
    INSTALL_RPATH "${_RPATH_BASE}")
endif()

# macOS: use @loader_path to keep the binary relocatable
if(APPLE)
  set_target_properties(gzstd PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path")
endif()

# =====================================================
#                    Summary
# =====================================================
message(STATUS "\n==== gzstd configure summary ====")
message(STATUS " CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")
message(STATUS " BUILD_STATIC     : ${BUILD_STATIC}")
if(HAVE_NVCOMP)
  message(STATUS " GPU backend     : ENABLED (CUDA ${CUDAToolkit_VERSION})")
  message(STATUS " nvCOMP include  : ${NVCOMP_INCLUDE_DIR}")
  message(STATUS " nvCOMP library  : ${NVCOMP_LIBRARY}")
  if(BUILD_STATIC)
    message(STATUS " CUDA runtime    : static (cudart_static)")
  else()
    message(STATUS " CUDA runtime    : shared (cudart)")
  endif()
  if(HAVE_NVML)
    message(STATUS " NVML            : runtime dlopen (driver optional; no DT_NEEDED)")
  else()
    message(STATUS " NVML            : disabled (free-VRAM fallback)")
  endif()
else()
  message(STATUS " GPU backend     : DISABLED")
endif()
message(STATUS " Zstd            : target ZSTD::ZSTD")
message(STATUS " C++ Standard    : ${CMAKE_CXX_STANDARD}")
if(BUILD_STATIC)
  message(STATUS " Linking         : static (portable binary)")
  if(NVCOMP_IS_SHARED)
    message(STATUS " Runtime deps    : glibc, libnvcomp.so (NVIDIA driver optional)")
  else()
    message(STATUS " Runtime deps    : glibc only (NVIDIA driver optional)")
  endif()
else()
  message(STATUS " Linking         : shared (default)")
endif()
if(BUILD_STATIC AND NVCOMP_IS_SHARED)
  message(STATUS " NOTE: nvCOMP is shared; use 'make install' to bundle .so with binary")
endif()
message(STATUS "================================\n")

# =====================================================
#                    Install
# Installs the gzstd binary and bundles any shared libs
# that couldn't be linked statically (nvCOMP).  NVML is
# dlopen'd from the driver at runtime — never bundled.
# Usage: cmake --install build --prefix ./dist
# =====================================================
install(TARGETS gzstd RUNTIME DESTINATION bin)

# Bundle shared libraries that the binary needs at runtime
if(HAVE_NVCOMP AND BUILD_STATIC AND NVCOMP_IS_SHARED)
  # Resolve symlinks to find the real .so file, then install
  # both the real file and the soname symlink
  get_filename_component(_NVCOMP_REAL "${NVCOMP_LIBRARY}" REALPATH)
  get_filename_component(_NVCOMP_DIR  "${NVCOMP_LIBRARY}" DIRECTORY)
  get_filename_component(_NVCOMP_NAME "${NVCOMP_LIBRARY}" NAME)
  install(FILES "${_NVCOMP_REAL}" DESTINATION bin)
  # Create the soname symlink (e.g., libnvcomp.so.5 -> libnvcomp.so.5.0.1)
  get_filename_component(_NVCOMP_REAL_NAME "${_NVCOMP_REAL}" NAME)
  if(NOT _NVCOMP_NAME STREQUAL _NVCOMP_REAL_NAME)
    install(CODE "
      execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
        \"${_NVCOMP_REAL_NAME}\"
        \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/${_NVCOMP_NAME}\")
      message(STATUS \"Symlink: ${_NVCOMP_NAME} -> ${_NVCOMP_REAL_NAME}\")
    ")
  endif()
  # Also grab the unversioned .so symlink if it exists
  if(EXISTS "${_NVCOMP_DIR}/libnvcomp.so" AND NOT _NVCOMP_NAME STREQUAL "libnvcomp.so")
    install(CODE "
      execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
        \"${_NVCOMP_REAL_NAME}\"
        \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/libnvcomp.so\")
    ")
  endif()
endif()
