--echo #
--echo # WL#Vector: provider batch search thread limits
--echo #

eval SELECT VEC_INDEX_CREATE('$vector_index_name', 2, 'EUCLIDEAN',
                             '$vector_mode', '$vector_provider');
eval SELECT VEC_INDEX_UPSERT('$vector_index_name', 1, '[1,0]');
eval SELECT VEC_INDEX_UPSERT('$vector_index_name', 2, '[0,1]');

SET GLOBAL vector_batch_search_threads = 1;
eval SELECT VEC_INDEX_SEARCH_BATCH('$vector_index_name',
       CONCAT(VEC_FROMTEXT('[1,0]'), VEC_FROMTEXT('[0,1]')), 2, 2)
       AS threads_1;

SET GLOBAL vector_batch_search_threads = 32;
eval SELECT VEC_INDEX_SEARCH_BATCH('$vector_index_name',
       CONCAT(VEC_FROMTEXT('[1,0]'), VEC_FROMTEXT('[0,1]')), 2, 2)
       AS threads_32;

eval SELECT VEC_INDEX_SEARCH_BATCH('$vector_index_name',
       CONCAT(VEC_FROMTEXT('[1,0]'), VEC_FROMTEXT('[0,1]')), 2, 2)
       AS provider_override_1;

eval SELECT VEC_INDEX_DROP('$vector_index_name');
