# MySQL Vector 中文入口

MySQL Vector 是向量能力扩展，用于在 MySQL 内保存向量、构建向量索引，并通过 SQL 完成语义检索、推荐召回、多模态相似度匹配等工作负载。

[完整 HTML 手册](README_VECTOR.html)收录了函数、参数、DDL、状态项、运行模式、导入方式、构建模式、后端说明和性能测试结果。

## 支持版本

| MySQL 版本 | MySQL 分支 | 完整手册 |
|------------|------------|----------|
| MySQL 8.0.46 | [mysql-vector-8046](../../tree/mysql-vector-8046) | [README_VECTOR.html](README_VECTOR.html) |

## 编译示例

下面的示例假设 FAISS、HNSWLib 和 DiskANN 源码位于 `../deps/`。MySQL-Vector 使用原生运行时调度这些后端库；源码构建、预编译库、无向量构建和后端回退说明见 HTML 手册。

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

## SQL 快速示例

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

## 相关文档

- [完整 HTML 手册（中文/English 可切换）](README_VECTOR.html)
- [English quick start](README_VECTOR_EN.md)

## 致谢

MySQL Vector 参考了 [AliSQL](https://github.com/alibaba/AliSQL)、[MariaDB](https://github.com/MariaDB/server) 和 [Milvus](https://github.com/milvus-io/milvus) 的公开代码与工程思路，并直接集成 [hnswlib](https://github.com/nmslib/hnswlib)、[Faiss](https://github.com/facebookresearch/faiss) 和 [DiskANN](https://github.com/microsoft/DiskANN)。
