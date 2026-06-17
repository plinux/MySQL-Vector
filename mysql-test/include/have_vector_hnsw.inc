--source include/have_vector_index.inc

if (`SELECT COUNT(*) = 0 FROM INFORMATION_SCHEMA.VECTOR_LIBRARIES WHERE LIBRARY = 'hnsw' AND SUPPORT = 'YES'`)
{
  --skip Test requires compiled hnswlib vector backend.
}
