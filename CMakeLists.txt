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
    find_library(ZSTD_LIBRARY NAMES ${_ZSTD_LIB_NAMES}
                 HINTS ${ZSTD_ROOT} ${ZSTD_PREFIX}
                       ENV CONDA_PREFIX ENV ZSTD_ROOT ENV ZSTD_PREFIX
                 PATH_SUFFIXES ${_ZSTD_LIB_SUFFIXES})
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
    find_path(NVCOMP_INCLUDE_DIR
      NAMES nvcomp/zstd.h
      HINTS $ENV{NVCOMP_ROOT} $ENV{CONDA_PREFIX}
      PATH_SUFFIXES include)

    if(BUILD_STATIC)
      # Prefer libnvcomp_static.a; fall back to shared if user only has
      # conda-forge's package (which is shared-only).  NVIDIA's official
      # tarball at https://developer.download.nvidia.com/compute/nvcomp/redist/
      # ships libnvcomp_static.a.  See BUILD.md for the recipe.
      find_library(NVCOMP_LIBRARY
        NAMES nvcomp_static nvcomp
        HINTS $ENV{NVCOMP_ROOT} $ENV{CONDA_PREFIX}
        PATH_SUFFIXES lib lib64)
    else()
      find_library(NVCOMP_LIBRARY
        NAMES nvcomp
        HINTS $ENV{NVCOMP_ROOT} $ENV{CONDA_PREFIX}
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

# =====================================================
#          NVML (GPU utilization monitoring)
# When BUILD_STATIC is ON, use the NVML stubs library
# (links at build time, resolves against driver at runtime).
# NVML cannot be fully statically linked  it's part of
# the NVIDIA driver.  The stubs library is always used.
# =====================================================
set(HAVE_NVML FALSE)
if(HAVE_NVCOMP AND CUDAToolkit_FOUND)
  find_library(NVML_LIBRARY
    NAMES nvidia-ml
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          "/usr/lib/x86_64-linux-gnu"
          "/usr/lib64"
          "/usr/local/cuda/lib64/stubs"
          "/usr/lib/x86_64-linux-gnu/nvidia/current"
    PATH_SUFFIXES stubs)

  if(NVML_LIBRARY)
    set(HAVE_NVML TRUE)
    message(STATUS "NVML found: ${NVML_LIBRARY}")
  else()
    message(STATUS "NVML (nvidia-ml) not found; GPU selection will use free-VRAM fallback")
  endif()
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
endif()
if(HAVE_NVML)
  target_compile_definitions(gzstd PRIVATE HAVE_NVML)
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

  # Static CUDA runtime needs dl and rt on Linux
  if(BUILD_STATIC AND UNIX)
    target_link_libraries(gzstd PRIVATE dl rt)
  endif()

  if(HAVE_NVML)
    target_link_libraries(gzstd PRIVATE ${NVML_LIBRARY})
  endif()
else()
  target_link_libraries(gzstd PRIVATE
    ZSTD::ZSTD
    Threads::Threads)
endif()

# Static linking: prefer static libstdc++ and libgcc for maximum portability.
# The resulting binary only needs glibc (2.17+) and the NVIDIA driver at runtime.
if(BUILD_STATIC AND NOT MSVC)
  target_link_options(gzstd PRIVATE
    -static-libgcc
    -static-libstdc++)
endif()

# =====================================================
#                     RPATH
# Make ./build/gzstd run cleanly with Conda/system libs.
# When BUILD_STATIC is ON, RPATH is less important since
# most dependencies are linked statically.  We still set
# it for NVML (always shared  part of the driver).
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
    message(STATUS " NVML            : ${NVML_LIBRARY} (always shared  driver component)")
  else()
    message(STATUS " NVML            : not found (free-VRAM fallback)")
  endif()
else()
  message(STATUS " GPU backend     : DISABLED")
endif()
message(STATUS " Zstd            : target ZSTD::ZSTD")
message(STATUS " C++ Standard    : ${CMAKE_CXX_STANDARD}")
if(BUILD_STATIC)
  message(STATUS " Linking         : static (portable binary)")
  if(NVCOMP_IS_SHARED)
    message(STATUS " Runtime deps    : glibc, NVIDIA driver, libnvcomp.so")
  else()
    message(STATUS " Runtime deps    : glibc, NVIDIA driver (libnvidia-ml.so)")
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
# that couldn't be linked statically (nvCOMP, NVML).
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
