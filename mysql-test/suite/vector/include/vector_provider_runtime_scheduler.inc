--echo #
--echo # WL#Vector: provider runtime scheduler diagnostics
--echo #

eval SELECT VEC_INDEX_CREATE('$vector_index_name', 2, 'EUCLIDEAN',
                             '$vector_mode', '$vector_provider');
eval SELECT VEC_INDEX_UPSERT('$vector_index_name', 1,
                             VEC_FROMTEXT('[1,0]'));
eval SELECT VEC_INDEX_UPSERT('$vector_index_name', 2,
                             VEC_FROMTEXT('[2,0]'));
eval SELECT VEC_INDEX_REBUILD('$vector_index_name');

--replace_column 1 <runtime>
eval SELECT JSON_UNQUOTE(JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'),
              '$.backend_build_runtime')) AS runtime,
            JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'),
              '$.backend_build_effective_threads') >= 1
              AS effective_threads_present,
            JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'),
              '$.backend_build_effective_blas_threads') >= 1
              AS effective_blas_threads_present,
            COALESCE(JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'),
              '$.backend_build_pq_chunks'), 0) AS pq_chunks,
            COALESCE(JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'),
              '$.backend_build_cache_nodes'), 0) AS cache_nodes;

--replace_column 3 <value>
eval SHOW VECTOR STATUS FOR INDEX $vector_index_name
     WHERE Section = 'BACKEND_HEALTH'
       AND Variable_name LIKE '$vector_index_name.scheduler_%';

eval SELECT VEC_INDEX_DROP('$vector_index_name');
