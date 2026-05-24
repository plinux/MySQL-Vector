# MySQL Vector 中文说明

MySQL Vector 是向量能力扩展，用于在 MySQL 内保存向量、构建向量索引，并通过 SQL 完成语义检索、推荐召回、多模态相似度匹配等工作负载。

## 支持版本

| MySQL 版本 | MySQL 分支 | HNSW版本 | FAISS版本 | DiskANN 版本 | DiskANN补丁路径 |
|------------|------------|----------|-----------|--------------|----------------|
| MySQL 8.0.46 | [mysql-vector-8046](../../tree/mysql-vector-8046) | [v0.9.0](https://github.com/nmslib/hnswlib/commit/d9b3608c83d83b46c96e25088cb1d729b29dcfe9) | [v1.14.2-5-gca87f41df](https://github.com/facebookresearch/faiss/commit/ca87f41dfe58336245a013118e38ddaad0f7cab3) | [diskann-garnet-v1.0.27](https://github.com/microsoft/DiskANN/commit/e3139b4838f20f1ffa36a600ef1fce403e68f92f) | [extra/vector/diskann-garnet-bulk-build-c-abi.patch](extra/vector/diskann-garnet-bulk-build-c-abi.patch) |

测试套件和三方软件需要适配 MySQL-Vector 接口才能使用，已经适配的版本如下，可以提取补丁使用。

| 测试套件 | 软件版本 / 基线 | 已验证分支 | 已验证提交 |
|----------|-----------------|------------|------------|
| [ANN-Benchmarks](https://github.com/plinux/ann-benchmarks) | `f402b2c` | [mysql-vector-connector-f402b2c](https://github.com/plinux/ann-benchmarks/tree/mysql-vector-connector-f402b2c) | `87cc631596291fa69d41fd4ad431211fa5f92e12` |
| [VectorDBBench](https://github.com/plinux/VectorDBBench) | `v1.0.22` | [mysql-vector-connector-1.0.22](https://github.com/plinux/VectorDBBench/tree/mysql-vector-connector-1.0.22) | `9672bbfccf4579f716dac56f0882acf290dd01df` |
| [Mem0](https://github.com/plinux/mem0) | `v2.0.2` | [mysql-vector-connector-2.0.2](https://github.com/plinux/mem0/tree/mysql-vector-connector-2.0.2) | `ad9d2f5e76c51015c6f2a2c610bf4669ce9d4278` |

## 项目作用

- 在 InnoDB 表中使用 `VECTOR(dim)` 保存向量。
- 支持 `VEC_FROMTEXT()`、`VEC_DISTANCE()`、`VEC_INDEX_SEARCH()` 等 SQL 函数。
- 支持命名向量索引和 `CREATE VECTOR INDEX` 表级索引。
- 支持 HNSWLIB、Faiss 和 DiskANN 后端；DiskANN 官方 ABI 默认串行构建，可选实验 bulk ABI patch。
- 支持 truth-store 持久化、事务提交、恢复、复制回归和状态观测。

## 构建概览

| 项目 | 当前说明 |
|------|----------|
| 编译开关 | `-DHAVE_VECTOR_INDEX=ON` 开启向量；`OFF` 构建标准 MySQL 行为版本 |
| 默认后端 | HNSWLIB + Faiss 源码原生编译；DiskANN 通过 Garnet 动态库运行时加载 |
| 主要平台 | Linux、macOS |
| 文档语言 | 中文入口、英文入口、单一 HTML 完整手册 |

## 推荐上手路径

1. 先用 Debug 或 Release 构建一个 `HAVE_VECTOR_INDEX=ON` 的二进制。
2. 启动独立测试实例，执行下面的快速 SQL smoke。
3. 阅读完整 HTML 手册了解构建参数、函数、DDL、状态项和 DiskANN 回退策略。
4. 做性能测试前，先确认 `SHOW GLOBAL STATUS LIKE 'Vector_%'` 和 `SHOW VECTOR STATUS` 输出正常。

## 快速构建

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

如果只想验证标准 MySQL 行为版本，使用 `-DHAVE_VECTOR_INDEX=OFF`，并且不要传向量后端路径参数。

## 快速 SQL 示例

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

## 完整文档

- [完整 HTML 手册（中文/English 可切换）](README_VECTOR.html)
- [English quick start](README_VECTOR_EN.md)

完整 HTML 手册包含：构建参数、运行时变量、升级保护、向量类型、距离枚举、函数表、命名索引、事务函数、表级 DDL、状态观测、DiskANN 串行回退和实验 bulk ABI patch 说明。
