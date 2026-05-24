# MySQL Vector English Guide

MySQL Vector is a vector-search extension. It lets MySQL store vector values, build vector indexes, and run similarity search through SQL for semantic search, recommendation, retrieval-augmented generation, and multimodal workloads.

## Supported Versions

| MySQL version | MySQL branch | HNSW version | FAISS version | DiskANN version | DiskANN patch path |
|---------------|--------------|--------------|---------------|-----------------|--------------------|
| MySQL 8.0.46 | [mysql-vector-8046](../../tree/mysql-vector-8046) | [v0.9.0](https://github.com/nmslib/hnswlib/commit/d9b3608c83d83b46c96e25088cb1d729b29dcfe9) | [v1.14.2-5-gca87f41df](https://github.com/facebookresearch/faiss/commit/ca87f41dfe58336245a013118e38ddaad0f7cab3) | [diskann-garnet-v1.0.27](https://github.com/microsoft/DiskANN/commit/e3139b4838f20f1ffa36a600ef1fce403e68f92f) | [extra/vector/diskann-garnet-bulk-build-c-abi.patch](extra/vector/diskann-garnet-bulk-build-c-abi.patch) |

Test suites and third-party software need MySQL-Vector interface adapters before use. The adapted versions are listed below, and their patches can be extracted from these branches.

| Test suite | Software version / baseline | Verified branch | Verified commit |
|------------|-----------------------------|-----------------|-----------------|
| [ANN-Benchmarks](https://github.com/plinux/ann-benchmarks) | `f402b2c` | [mysql-vector-connector-f402b2c](https://github.com/plinux/ann-benchmarks/tree/mysql-vector-connector-f402b2c) | `87cc631596291fa69d41fd4ad431211fa5f92e12` |
| [VectorDBBench](https://github.com/plinux/VectorDBBench) | `v1.0.22` | [mysql-vector-connector-1.0.22](https://github.com/plinux/VectorDBBench/tree/mysql-vector-connector-1.0.22) | `9672bbfccf4579f716dac56f0882acf290dd01df` |
| [Mem0](https://github.com/plinux/mem0) | `v2.0.2` | [mysql-vector-connector-2.0.2](https://github.com/plinux/mem0/tree/mysql-vector-connector-2.0.2) | `ad9d2f5e76c51015c6f2a2c610bf4669ce9d4278` |

## What It Provides

- `VECTOR(dim)` columns in InnoDB tables.
- SQL functions such as `VEC_FROMTEXT()`, `VEC_DISTANCE()`, and `VEC_INDEX_SEARCH()`.
- Standalone named vector indexes and table-bound `CREATE VECTOR INDEX` syntax.
- HNSWLIB, Faiss, and DiskANN backends. DiskANN uses the official serial Garnet ABI by default and can use an experimental bulk-build ABI patch when explicitly applied.
- Durable truth-store persistence, transaction integration, recovery, replication coverage, and observability surfaces.

## Build Overview

| Item | Current value |
|------|---------------|
| Build gate | `-DHAVE_VECTOR_INDEX=ON` enables vector support; `OFF` builds an upstream-compatible MySQL behavior profile |
| Default native backends | HNSWLIB and Faiss from source; DiskANN through a runtime-loaded Garnet shared library |
| Main platforms | Linux and macOS |
| Documentation | Chinese quick start, English quick start, and one bilingual HTML manual |

## Recommended First Steps

1. Build a Debug or Release binary with `HAVE_VECTOR_INDEX=ON`.
2. Start an isolated test instance and run the quick SQL smoke below.
3. Read the full HTML manual for build options, SQL functions, DDL, status output, and DiskANN fallback behavior.
4. Before benchmarking, verify `SHOW GLOBAL STATUS LIKE 'Vector_%'` and `SHOW VECTOR STATUS` output.

## Quick Build

```bash
BUILD_ROOT=../build/mysql-vector
BOOST_ROOT="${BUILD_ROOT}/deps"
FAISS_ROOT=../deps/FAISS
HNSWLIB_ROOT=../deps/HNSWLib
DISKANN_LIB_DIR=../deps/DiskANN/target/release

cmake -S . -B "${BUILD_ROOT}/release" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_DEBUG=0 \
  -DHAVE_VECTOR_INDEX=ON \
  -DWITH_NDB=OFF \
  -DWITH_NDBCLUSTER_STORAGE_ENGINE=OFF \
  -DDOWNLOAD_BOOST=1 \
  -DWITH_BOOST="${BOOST_ROOT}" \
  -DMYSQL_VECTOR_FAISS_ROOT="${FAISS_ROOT}" \
  -DMYSQL_VECTOR_HNSWLIB_ROOT="${HNSWLIB_ROOT}" \
  -DMYSQL_VECTOR_DISKANN_LIB_DIR="${DISKANN_LIB_DIR}"

cmake --build "${BUILD_ROOT}/release" --parallel
```

To validate an upstream-compatible MySQL behavior profile, build with `-DHAVE_VECTOR_INDEX=OFF` and omit all vector backend path arguments.

## Quick SQL Smoke

```sql
CREATE TABLE items (
  id BIGINT PRIMARY KEY,
  embedding VECTOR(3)
) ENGINE=InnoDB;

INSERT INTO items VALUES
  (1, VEC_FROMTEXT('[1,2,3]')),
  (2, VEC_FROMTEXT('[4,5,6]'));

SELECT id, VEC_DISTANCE(embedding, VEC_FROMTEXT('[3,1,2]'), 'L2') AS distance
FROM items
ORDER BY distance
LIMIT 5;

SELECT VEC_INDEX_CREATE('idx_items', 3, 'EUCLIDEAN', 'MEMORY', 'HNSWLIB') AS created;
SELECT VEC_INDEX_UPSERT('idx_items', 1, VEC_FROMTEXT('[1,2,3]')) AS upserted;
SELECT VEC_INDEX_SEARCH_WITH_DISTANCE('idx_items', VEC_FROMTEXT('[3,1,2]'), 5) AS result;
```

## Full Documentation

- [Full bilingual HTML manual](README_VECTOR.html)
- [中文快速入口](README_VECTOR_CN.md)

The HTML manual covers build options, runtime variables, upgrade protection, vector types, distance metrics, function references, named indexes, transaction functions, table-level DDL, observability, and the DiskANN serial fallback / experimental bulk ABI patch policy.
