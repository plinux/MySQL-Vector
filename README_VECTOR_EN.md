# MySQL Vector English guide

MySQL Vector is a vector-search extension. It lets MySQL store vector values, build vector indexes, and run similarity search through SQL for semantic search, recommendation, retrieval, and multimodal workloads.

The [full HTML manual](README_VECTOR.html) documents all functions, parameters, DDL, status fields, run modes, import methods, build modes, backends, and performance results.

## Supported versions

| MySQL version | MySQL branch | Full manual |
|---------------|--------------|-------------|
| MySQL 8.0.46 | [mysql-vector-8046](../../tree/mysql-vector-8046) | [README_VECTOR.html](README_VECTOR.html) |

## Build example

The example below assumes FAISS, HNSWLib, and DiskANN source trees are under `../deps/`. MySQL-Vector uses the native runtime to dispatch work to these backend libraries. See the HTML manual for source builds, prebuilt libraries, no-vector builds, and backend fallback details.

```bash
BUILD_ROOT=../build/mysql-vector
BOOST_ROOT="${BUILD_ROOT}/deps"
FAISS_ROOT=../deps/FAISS
HNSWLIB_ROOT=../deps/HNSWLib
DISKANN_ROOT=../deps/DiskANN

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
  -DMYSQL_VECTOR_DISKANN_ROOT="${DISKANN_ROOT}"

cmake --build "${BUILD_ROOT}/release" --parallel
```

## Quick SQL example

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
```

## Related documentation

- [Full bilingual HTML manual](README_VECTOR.html)
- [中文快速入口](README_VECTOR_CN.md)

## Acknowledgements

MySQL Vector draws on public code and engineering ideas from [AliSQL](https://github.com/alibaba/AliSQL), [MariaDB](https://github.com/MariaDB/server), and [Milvus](https://github.com/milvus-io/milvus). It directly integrates [hnswlib](https://github.com/nmslib/hnswlib), [Faiss](https://github.com/facebookresearch/faiss), and [DiskANN](https://github.com/microsoft/DiskANN).
