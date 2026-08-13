--echo #
--echo # Provider-gated create, rebuild, search, restart, and drop smoke
--echo #

DROP DATABASE IF EXISTS vector_provider_restart;
CREATE DATABASE vector_provider_restart;
USE vector_provider_restart;

--eval SELECT VEC_INDEX_CREATE('$vector_restart_index', 2, 'EUCLIDEAN', '$vector_restart_mode', '$vector_restart_provider', 0, 'standalone')
--eval SELECT VEC_INDEX_UPSERT('$vector_restart_index', 1, '[1,0]')
--eval SELECT VEC_INDEX_UPSERT('$vector_restart_index', 2, '[0,1]')
--eval SELECT VEC_INDEX_REBUILD('$vector_restart_index')
--eval SELECT JSON_UNQUOTE(JSON_EXTRACT(VEC_INDEX_INFO('$vector_restart_index'), '$.provider')) AS provider
--eval SELECT JSON_UNQUOTE(JSON_EXTRACT(VEC_INDEX_INFO('$vector_restart_index'), '$.owner_schema')) AS owner_schema
--eval SELECT VEC_INDEX_SEARCH('$vector_restart_index', '[1,0]', 1)

--source include/restart_mysqld.inc

USE vector_provider_restart;
--eval SELECT JSON_UNQUOTE(JSON_EXTRACT(VEC_INDEX_INFO('$vector_restart_index'), '$.provider')) AS provider
--eval SELECT JSON_UNQUOTE(JSON_EXTRACT(VEC_INDEX_INFO('$vector_restart_index'), '$.owner_schema')) AS owner_schema
--eval SELECT VEC_INDEX_SEARCH('$vector_restart_index', '[1,0]', 1)
--eval SELECT VEC_INDEX_DROP('$vector_restart_index')

USE test;
DROP DATABASE vector_provider_restart;
