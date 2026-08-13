--echo #
--echo # WL#Vector: provider HNSW build parameter persistence
--echo #

eval SELECT VEC_INDEX_CREATE('$vector_index_name', 2, 'COSINE',
                             '$vector_mode', '$vector_provider');
eval SELECT VEC_INDEX_SET_HNSW_BUILD_PARAMS('$vector_index_name', 24, 320);

eval SELECT JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'), '$.hnsw_m')
         AS hnsw_m,
       JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'),
         '$.hnsw_ef_construction') AS hnsw_ef_construction;
eval SHOW VECTOR STATUS FOR INDEX $vector_index_name
     WHERE Section = 'INDEX_STATE'
       AND Variable_name LIKE '$vector_index_name.hnsw_%';

--source include/restart_mysqld.inc

eval SELECT JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'), '$.hnsw_m')
         AS hnsw_m,
       JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'),
         '$.hnsw_ef_construction') AS hnsw_ef_construction;
eval SHOW VECTOR STATUS FOR INDEX $vector_index_name
     WHERE Section = 'INDEX_STATE'
       AND Variable_name LIKE '$vector_index_name.hnsw_%';

eval SELECT VEC_INDEX_DROP('$vector_index_name');
