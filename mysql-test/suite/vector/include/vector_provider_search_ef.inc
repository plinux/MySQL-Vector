--echo #
--echo # WL#Vector: provider search ef persistence
--echo #

eval SELECT VEC_INDEX_CREATE('$vector_index_name', 2, 'COSINE',
                             '$vector_mode', '$vector_provider');
eval SELECT VEC_INDEX_SET_SEARCH_EF('$vector_index_name', $vector_search_ef);

eval SELECT JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'), '$.search_ef')
       AS search_ef;
eval SHOW VECTOR STATUS FOR INDEX $vector_index_name
     WHERE Section = 'INDEX_STATE'
       AND Variable_name = '$vector_index_name.search_ef';

--source include/restart_mysqld.inc

eval SELECT JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'), '$.search_ef')
       AS search_ef;
eval SHOW VECTOR STATUS FOR INDEX $vector_index_name
     WHERE Section = 'INDEX_STATE'
       AND Variable_name = '$vector_index_name.search_ef';

eval SELECT VEC_INDEX_DROP('$vector_index_name');
