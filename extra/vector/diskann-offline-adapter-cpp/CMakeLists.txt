cmake_minimum_required(VERSION 3.16)

project(mysql_vector_diskann_offline_adapter LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MYSQL_VECTOR_DISKANN_ROOT "$ENV{MYSQL_VECTOR_DISKANN_ROOT}" CACHE PATH
    "Root directory of the official DiskANN C++ source tree")
set(MYSQL_VECTOR_DISKANN_INCLUDES "$ENV{MYSQL_VECTOR_DISKANN_INCLUDES}"
    CACHE PATH "Directory containing DiskANN C++ public headers")
set(MYSQL_VECTOR_DISKANN_LIB_DIR "$ENV{MYSQL_VECTOR_DISKANN_LIB_DIR}"
    CACHE PATH "Directory containing the official DiskANN C++ library")
option(MYSQL_VECTOR_DISKANN_STATIC
       "Link the adapter with a static official DiskANN C++ library" OFF)

if(NOT MYSQL_VECTOR_DISKANN_INCLUDES AND MYSQL_VECTOR_DISKANN_ROOT)
  if(EXISTS "${MYSQL_VECTOR_DISKANN_ROOT}/include/pq_flash_index.h")
    set(MYSQL_VECTOR_DISKANN_INCLUDES
        "${MYSQL_VECTOR_DISKANN_ROOT}/include")
  endif()
endif()

if(NOT MYSQL_VECTOR_DISKANN_INCLUDES AND MYSQL_VECTOR_DISKANN_LIB_DIR)
  foreach(DISKANN_INCLUDE_CANDIDATE
      "${MYSQL_VECTOR_DISKANN_LIB_DIR}/../include"
      "${MYSQL_VECTOR_DISKANN_LIB_DIR}/../../include"
      "${MYSQL_VECTOR_DISKANN_LIB_DIR}/include")
    if(EXISTS "${DISKANN_INCLUDE_CANDIDATE}/pq_flash_index.h")
      set(MYSQL_VECTOR_DISKANN_INCLUDES "${DISKANN_INCLUDE_CANDIDATE}")
      break()
    endif()
  endforeach()
endif()

if(NOT MYSQL_VECTOR_DISKANN_INCLUDES)
  find_path(MYSQL_VECTOR_DISKANN_INCLUDES
    NAMES pq_flash_index.h)
endif()

if(NOT MYSQL_VECTOR_DISKANN_INCLUDES OR
   NOT EXISTS "${MYSQL_VECTOR_DISKANN_INCLUDES}/pq_flash_index.h")
  message(FATAL_ERROR
      "MYSQL_VECTOR_DISKANN_INCLUDES must contain the official DiskANN "
      "C++ headers, for example pq_flash_index.h")
endif()

set(MYSQL_VECTOR_SAVED_FIND_LIBRARY_SUFFIXES
    "${CMAKE_FIND_LIBRARY_SUFFIXES}")
if(MYSQL_VECTOR_DISKANN_STATIC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

find_library(MYSQL_VECTOR_DISKANN_LIBRARY
  NAMES diskann libdiskann
  HINTS ${MYSQL_VECTOR_DISKANN_LIB_DIR}
  PATH_SUFFIXES lib lib64 src)

set(CMAKE_FIND_LIBRARY_SUFFIXES
    "${MYSQL_VECTOR_SAVED_FIND_LIBRARY_SUFFIXES}")

set(MYSQL_VECTOR_DISKANN_LIBRARY_TARGET "")
find_package(OpenMP)
if(NOT MYSQL_VECTOR_DISKANN_LIBRARY AND MYSQL_VECTOR_DISKANN_ROOT AND
   EXISTS "${MYSQL_VECTOR_DISKANN_ROOT}/CMakeLists.txt")
  include(ExternalProject)

  set(MYSQL_VECTOR_DISKANN_SOURCE_BUILD_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/diskann-cpp-build")
  if(MYSQL_VECTOR_DISKANN_STATIC)
    set(MYSQL_VECTOR_DISKANN_SOURCE_LIBRARY
        "${MYSQL_VECTOR_DISKANN_SOURCE_BUILD_DIR}/src/${CMAKE_STATIC_LIBRARY_PREFIX}diskann${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(MYSQL_VECTOR_DISKANN_BUILD_SHARED_LIBS OFF)
  else()
    set(MYSQL_VECTOR_DISKANN_SOURCE_LIBRARY
        "${MYSQL_VECTOR_DISKANN_SOURCE_BUILD_DIR}/src/${CMAKE_SHARED_LIBRARY_PREFIX}diskann${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(MYSQL_VECTOR_DISKANN_BUILD_SHARED_LIBS ON)
  endif()
  set(MYSQL_VECTOR_DISKANN_SOURCE_TARGET diskann)
  # cpp_main selects its own compiler, so it must also probe matching OpenMP
  # flags instead of inheriting the parent toolchain's cached results.
  set(MYSQL_VECTOR_DISKANN_DEPENDENCY_CMAKE_ARGS "")
  set(MYSQL_VECTOR_DISKANN_TOOLCHAIN_CMAKE_ARGS "")
  if(NOT MSVC)
    find_program(MYSQL_VECTOR_DISKANN_C_COMPILER NAMES gcc)
    find_program(MYSQL_VECTOR_DISKANN_CXX_COMPILER NAMES g++)
    if(NOT MYSQL_VECTOR_DISKANN_C_COMPILER OR
       NOT MYSQL_VECTOR_DISKANN_CXX_COMPILER)
      message(FATAL_ERROR
          "DiskANN cpp_main source builds require gcc and g++")
    endif()
    list(APPEND MYSQL_VECTOR_DISKANN_TOOLCHAIN_CMAKE_ARGS
      "-DCMAKE_C_COMPILER=${MYSQL_VECTOR_DISKANN_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${MYSQL_VECTOR_DISKANN_CXX_COMPILER}")
  endif()
  foreach(MYSQL_VECTOR_DISKANN_UPSTREAM_CACHE_VAR
      OMP_PATH
      MKL_PATH
      MKL_INCLUDE_PATH)
    if(DEFINED ${MYSQL_VECTOR_DISKANN_UPSTREAM_CACHE_VAR} AND
       NOT "${${MYSQL_VECTOR_DISKANN_UPSTREAM_CACHE_VAR}}" STREQUAL "")
      list(APPEND MYSQL_VECTOR_DISKANN_DEPENDENCY_CMAKE_ARGS
        "-D${MYSQL_VECTOR_DISKANN_UPSTREAM_CACHE_VAR}=${${MYSQL_VECTOR_DISKANN_UPSTREAM_CACHE_VAR}}")
    endif()
  endforeach()

  ExternalProject_Add(mysql_vector_diskann_cpp_project
      SOURCE_DIR "${MYSQL_VECTOR_DISKANN_ROOT}"
      BINARY_DIR "${MYSQL_VECTOR_DISKANN_SOURCE_BUILD_DIR}"
      CMAKE_ARGS
        ${MYSQL_VECTOR_DISKANN_TOOLCHAIN_CMAKE_ARGS}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=${MYSQL_VECTOR_DISKANN_BUILD_SHARED_LIBS}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        ${MYSQL_VECTOR_DISKANN_DEPENDENCY_CMAKE_ARGS}
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target
        ${MYSQL_VECTOR_DISKANN_SOURCE_TARGET}
      BUILD_BYPRODUCTS "${MYSQL_VECTOR_DISKANN_SOURCE_LIBRARY}"
      INSTALL_COMMAND "")

  if(MYSQL_VECTOR_DISKANN_STATIC)
    add_library(mysql_vector_diskann_cpp_library STATIC IMPORTED GLOBAL)
  else()
    add_library(mysql_vector_diskann_cpp_library SHARED IMPORTED GLOBAL)
  endif()
  set_target_properties(mysql_vector_diskann_cpp_library PROPERTIES
      IMPORTED_LOCATION "${MYSQL_VECTOR_DISKANN_SOURCE_LIBRARY}")
  add_dependencies(mysql_vector_diskann_cpp_library
      mysql_vector_diskann_cpp_project)
  set(MYSQL_VECTOR_DISKANN_LIBRARY_TARGET
      mysql_vector_diskann_cpp_library)
  set(MYSQL_VECTOR_DISKANN_LIBRARY
      "${MYSQL_VECTOR_DISKANN_SOURCE_LIBRARY}")
endif()

if(NOT MYSQL_VECTOR_DISKANN_LIBRARY AND
   NOT MYSQL_VECTOR_DISKANN_LIBRARY_TARGET)
  message(FATAL_ERROR
      "MYSQL_VECTOR_DISKANN_ROOT or MYSQL_VECTOR_DISKANN_LIB_DIR must point "
      "to a buildable official DiskANN C++ source tree or a built "
      "libdiskann library")
endif()

find_library(MYSQL_VECTOR_AIO_LIBRARY NAMES aio)
find_library(MYSQL_VECTOR_ATOMIC_LIBRARY NAMES atomic libatomic.so.1)

set(MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES "")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(MYSQL_VECTOR_DISKANN_MKL_LIBRARY_DIRS
      /opt/intel/oneapi/mkl/latest/lib/intel64
      /usr/lib/x86_64-linux-gnu
      /opt/intel/mkl/lib/intel64)
  set(MYSQL_VECTOR_DISKANN_MKL_INCLUDE_DIRS
      /opt/intel/oneapi/mkl/latest/include
      /usr/include/mkl
      /opt/intel/mkl/include)

  unset(MYSQL_VECTOR_MKL_ILP64_LIBRARY CACHE)
  unset(MYSQL_VECTOR_MKL_CORE_LIBRARY CACHE)
  unset(MYSQL_VECTOR_MKL_GNU_THREAD_LIBRARY CACHE)
  unset(MYSQL_VECTOR_MKL_SEQUENTIAL_LIBRARY CACHE)
  unset(MYSQL_VECTOR_MKL_ILP64_LIBRARY)
  unset(MYSQL_VECTOR_MKL_CORE_LIBRARY)
  unset(MYSQL_VECTOR_MKL_GNU_THREAD_LIBRARY)
  unset(MYSQL_VECTOR_MKL_SEQUENTIAL_LIBRARY)
  set(MYSQL_VECTOR_SAVED_FIND_LIBRARY_SUFFIXES
      "${CMAKE_FIND_LIBRARY_SUFFIXES}")
  if(MYSQL_VECTOR_DISKANN_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  endif()
  find_library(MYSQL_VECTOR_MKL_ILP64_LIBRARY
    NAMES mkl_intel_ilp64
    PATHS ${MYSQL_VECTOR_DISKANN_MKL_LIBRARY_DIRS})
  find_library(MYSQL_VECTOR_MKL_CORE_LIBRARY
    NAMES mkl_core
    PATHS ${MYSQL_VECTOR_DISKANN_MKL_LIBRARY_DIRS})
  find_library(MYSQL_VECTOR_MKL_GNU_THREAD_LIBRARY
    NAMES mkl_gnu_thread
    PATHS ${MYSQL_VECTOR_DISKANN_MKL_LIBRARY_DIRS})
  find_library(MYSQL_VECTOR_MKL_SEQUENTIAL_LIBRARY
    NAMES mkl_sequential
    PATHS ${MYSQL_VECTOR_DISKANN_MKL_LIBRARY_DIRS})
  set(CMAKE_FIND_LIBRARY_SUFFIXES
      "${MYSQL_VECTOR_SAVED_FIND_LIBRARY_SUFFIXES}")
  find_path(MYSQL_VECTOR_MKL_INCLUDE_DIR
    NAMES mkl.h
    PATHS ${MYSQL_VECTOR_DISKANN_MKL_INCLUDE_DIRS})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   MYSQL_VECTOR_MKL_ILP64_LIBRARY AND
   MYSQL_VECTOR_MKL_CORE_LIBRARY AND
   MYSQL_VECTOR_MKL_INCLUDE_DIR)
  if(MYSQL_VECTOR_MKL_SEQUENTIAL_LIBRARY)
    list(APPEND MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES
      "-Wl,--start-group"
      "${MYSQL_VECTOR_MKL_ILP64_LIBRARY}"
      "${MYSQL_VECTOR_MKL_SEQUENTIAL_LIBRARY}"
      "${MYSQL_VECTOR_MKL_CORE_LIBRARY}"
      "-Wl,--end-group")
  elseif(MYSQL_VECTOR_MKL_GNU_THREAD_LIBRARY)
    list(APPEND MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES
      "-Wl,--start-group"
      "${MYSQL_VECTOR_MKL_ILP64_LIBRARY}"
      "${MYSQL_VECTOR_MKL_GNU_THREAD_LIBRARY}"
      "${MYSQL_VECTOR_MKL_CORE_LIBRARY}"
      "-Wl,--end-group")
  else()
    message(FATAL_ERROR
      "DiskANN offline adapter found MKL headers and core libraries but "
      "neither libmkl_gnu_thread nor libmkl_sequential is available")
  endif()
else()
  find_package(BLAS)
  find_package(LAPACK)
  find_library(MYSQL_VECTOR_LAPACKE_LIBRARY NAMES lapacke)
  if(MYSQL_VECTOR_LAPACKE_LIBRARY)
    list(APPEND MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES
      "${MYSQL_VECTOR_LAPACKE_LIBRARY}")
  endif()
  if(LAPACK_FOUND)
    list(APPEND MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES
      ${LAPACK_LIBRARIES})
  elseif(BLAS_FOUND)
    list(APPEND MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES
      ${BLAS_LIBRARIES})
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT MYSQL_VECTOR_AIO_LIBRARY)
  message(FATAL_ERROR "libaio is required for the DiskANN offline adapter")
endif()

set(MYSQL_VECTOR_DISKANN_OFFLINE_ADAPTER_TYPE SHARED)
if(MYSQL_VECTOR_DISKANN_STATIC)
  set(MYSQL_VECTOR_DISKANN_OFFLINE_ADAPTER_TYPE STATIC)
endif()

add_library(mysql_vector_diskann_offline_adapter
  ${MYSQL_VECTOR_DISKANN_OFFLINE_ADAPTER_TYPE}
  mysql_vector_diskann_offline_adapter.cc)

target_include_directories(mysql_vector_diskann_offline_adapter PRIVATE
  "${MYSQL_VECTOR_DISKANN_INCLUDES}")

if(MYSQL_VECTOR_MKL_INCLUDE_DIR)
  target_include_directories(mysql_vector_diskann_offline_adapter SYSTEM
    PRIVATE "${MYSQL_VECTOR_MKL_INCLUDE_DIR}")
endif()

if(MYSQL_VECTOR_DISKANN_LIBRARY_TARGET)
  target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
    ${MYSQL_VECTOR_DISKANN_LIBRARY_TARGET})
else()
  target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
    "${MYSQL_VECTOR_DISKANN_LIBRARY}")
endif()

target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
  pthread
  ${CMAKE_DL_LIBS}
  m)

if(MYSQL_VECTOR_AIO_LIBRARY)
  target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
    "${MYSQL_VECTOR_AIO_LIBRARY}")
endif()

if(MYSQL_VECTOR_ATOMIC_LIBRARY)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
      "-Wl,--no-as-needed" "${MYSQL_VECTOR_ATOMIC_LIBRARY}" "-Wl,--as-needed")
  else()
    target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
      "${MYSQL_VECTOR_ATOMIC_LIBRARY}")
  endif()
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
    OpenMP::OpenMP_CXX)
endif()

if(MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
      "-Wl,--no-as-needed" ${MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES}
      "-Wl,--as-needed")
  else()
    target_link_libraries(mysql_vector_diskann_offline_adapter PRIVATE
      ${MYSQL_VECTOR_DISKANN_NUMERIC_LIBRARIES})
  endif()
endif()

set_target_properties(mysql_vector_diskann_offline_adapter PROPERTIES
  OUTPUT_NAME mysql_vector_diskann_offline_adapter)
