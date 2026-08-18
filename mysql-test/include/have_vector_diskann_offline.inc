--source include/have_vector_diskann.inc

if (`SELECT COUNT(*) = 0 FROM INFORMATION_SCHEMA.VECTOR_LIBRARIES WHERE LIBRARY = 'diskann' AND SUPPORT = 'YES' AND OFFLINE_BUILD = 'YES'`)
{
  --skip Test requires compiled DiskANN offline build support.
}
