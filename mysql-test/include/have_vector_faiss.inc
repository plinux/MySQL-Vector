--source include/have_vector_index.inc

if (`SELECT COUNT(*) = 0 FROM INFORMATION_SCHEMA.VECTOR_LIBRARIES WHERE LIBRARY = 'faiss' AND SUPPORT = 'YES'`)
{
  --skip Test requires compiled Faiss vector backend.
}
