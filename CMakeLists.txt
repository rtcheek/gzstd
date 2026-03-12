# CMakeLists.txt  gzstd (hybrid CPU+GPU Zstd) with robust Zstd fallback & RPATH hardening

cmake_minimum_required(VERSION 3.18)
project(gzstd LANGUAGES CXX)

# ---------------- Options ----------------
option(USE_NVCOMP "Enable nvCOMP (CUDA) GPU backend if found" ON)

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
# =====================================================

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
      target_link_libraries(ZSTD::ZSTD INTERFACE ${ZSTD_PKG_LINK_LIBRARIES})
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
    find_library(ZSTD_LIBRARY NAMES zstd
                 HINTS ${ZSTD_ROOT} ${ZSTD_PREFIX}
                       ENV CONDA_PREFIX ENV ZSTD_ROOT ENV ZSTD_PREFIX
                 PATH_SUFFIXES lib lib64)
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

    find_library(NVCOMP_LIBRARY
      NAMES nvcomp
      HINTS $ENV{NVCOMP_ROOT} $ENV{CONDA_PREFIX}
      PATH_SUFFIXES lib lib64)

    if(NVCOMP_INCLUDE_DIR AND NVCOMP_LIBRARY)
      set(HAVE_NVCOMP TRUE)
      message(STATUS "nvCOMP found: ${NVCOMP_LIBRARY}")
    else()
      message(STATUS "nvCOMP not found; building CPU-only (set NVCOMP_ROOT or CONDA_PREFIX if available)")
    endif()
  else()
    message(STATUS "CUDAToolkit not found; building CPU-only")
  endif()
endif()

# =====================================================
#          NVML (GPU utilization monitoring)
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

# Linkage
if(HAVE_NVCOMP)
  target_link_libraries(gzstd PRIVATE
    ZSTD::ZSTD
    Threads::Threads
    CUDA::cudart
    ${NVCOMP_LIBRARY})
  if(HAVE_NVML)
    target_link_libraries(gzstd PRIVATE ${NVML_LIBRARY})
  endif()
else()
  target_link_libraries(gzstd PRIVATE
    ZSTD::ZSTD
    Threads::Threads)
endif()

# =====================================================
#                     RPATH
# Make ./build/gzstd run cleanly with Conda/system libs.
# =====================================================
if(UNIX AND NOT APPLE)
  # Base RPATH entries near the binary
  set(_RPATH_BASE "$ORIGIN;$ORIGIN/..;$ORIGIN/../lib;$ORIGIN/../lib64")

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
if(HAVE_NVCOMP)
  message(STATUS " GPU backend     : ENABLED (CUDA ${CUDAToolkit_VERSION})")
  message(STATUS " nvCOMP include  : ${NVCOMP_INCLUDE_DIR}")
  message(STATUS " nvCOMP library  : ${NVCOMP_LIBRARY}")
  if(HAVE_NVML)
    message(STATUS " NVML            : ${NVML_LIBRARY}")
  else()
    message(STATUS " NVML            : not found (free-VRAM fallback)")
  endif()
else()
  message(STATUS " GPU backend     : DISABLED")
endif()
message(STATUS " Zstd            : target ZSTD::ZSTD")
message(STATUS " C++ Standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "================================\n")
