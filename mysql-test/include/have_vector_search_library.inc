--source include/have_vector_index.inc

if (`SELECT COUNT(*) = 0 FROM INFORMATION_SCHEMA.VECTOR_LIBRARIES WHERE SUPPORT = 'YES'`)
{
  --skip Test requires at least one compiled vector search library.
}
