--source include/have_vector_index.inc
--source include/have_debug.inc

if (`SELECT COUNT(*) = 0 FROM INFORMATION_SCHEMA.VECTOR_LIBRARIES WHERE SUPPORT = 'YES'`)
{
  --skip Test requires debug native provider support.
}
