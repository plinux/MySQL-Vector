#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_DIR="${SCRIPT_DIR}/diskann-offline-adapter-cpp"
BUILD_ROOT="${BUILD_ROOT:-${HOME}/Documents/BUILD}"
ADAPTER_BUILD_DIR="${ADAPTER_BUILD_DIR:-${BUILD_ROOT}/mysql-vector-diskann-offline-adapter}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

DISKANN_ROOT="${MYSQL_VECTOR_DISKANN_ROOT:-${DISKANN_ROOT:-}}"
DISKANN_INCLUDES="${MYSQL_VECTOR_DISKANN_INCLUDES:-${DISKANN_INCLUDES:-}}"
DISKANN_LIB_DIR="${MYSQL_VECTOR_DISKANN_LIB_DIR:-${DISKANN_LIB_DIR:-}}"
DISKANN_STATIC="${MYSQL_VECTOR_DISKANN_STATIC:-${DISKANN_STATIC:-OFF}}"

if [[ -z "${DISKANN_ROOT}" && -z "${DISKANN_INCLUDES}" && -z "${DISKANN_LIB_DIR}" ]]; then
  echo "MYSQL_VECTOR_DISKANN_ROOT, MYSQL_VECTOR_DISKANN_INCLUDES, or MYSQL_VECTOR_DISKANN_LIB_DIR must identify official DiskANN C++ inputs" >&2
  exit 2
fi

JOBS="${JOBS:-}"
if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
  else
    JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
  fi
fi

cmake -S "${SOURCE_DIR}" -B "${ADAPTER_BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DMYSQL_VECTOR_DISKANN_ROOT="${DISKANN_ROOT}" \
  -DMYSQL_VECTOR_DISKANN_INCLUDES="${DISKANN_INCLUDES}" \
  -DMYSQL_VECTOR_DISKANN_LIB_DIR="${DISKANN_LIB_DIR}" \
  -DMYSQL_VECTOR_DISKANN_STATIC="${DISKANN_STATIC}"

cmake --build "${ADAPTER_BUILD_DIR}" --target mysql_vector_diskann_offline_adapter \
  -j"${JOBS}"

echo "${ADAPTER_BUILD_DIR}"
