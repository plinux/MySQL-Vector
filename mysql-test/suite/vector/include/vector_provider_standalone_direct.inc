--echo #
--echo # WL#Vector: standalone provider through native runtime
--echo #

--disable_query_log
--let $vector_csv=$MYSQLTEST_VARDIR/$vector_csv_name
--let vector_csv=$vector_csv
--write_file $vector_csv
doc_id,vector
401,"[1,0]"
402,"[0,1]"
EOF
--enable_query_log

eval SELECT VEC_INDEX_CREATE('$vector_index_name', 2, 'EUCLIDEAN',
                             '$vector_mode', '$vector_provider', 0,
                             'standalone');

--disable_query_log
eval LOAD VECTOR DATA INFILE '$MYSQLTEST_VARDIR/$vector_csv_name'
  INTO VECTOR INDEX $vector_index_name
  FORMAT CSV
  REBUILD;
--enable_query_log

eval SELECT JSON_CONTAINS(
       VEC_INDEX_SEARCH('$vector_index_name', VEC_FROMTEXT('[1,0]'), 1),
       CAST(401 AS JSON)) AS first_doc_visible;
eval SELECT JSON_CONTAINS(
       VEC_INDEX_SEARCH('$vector_index_name', VEC_FROMTEXT('[0,1]'), 1),
       CAST(402 AS JSON)) AS second_doc_visible;
--replace_column 1 <backend_variant>
eval SELECT JSON_UNQUOTE(JSON_EXTRACT(VEC_INDEX_INFO('$vector_index_name'),
                                      '$.backend_variant'))
       AS backend_variant;

eval SELECT VEC_INDEX_DROP('$vector_index_name');

--remove_file $vector_csv
