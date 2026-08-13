--source include/have_vector_index.inc

--echo # Global - default
--eval SELECT @@global.$vector_build_threads_variable

--echo # Visible through SHOW VARIABLES and performance_schema.
--eval SHOW GLOBAL VARIABLES LIKE '$vector_build_threads_variable'
--eval SHOW SESSION VARIABLES LIKE '$vector_build_threads_variable'
--echo SELECT * FROM performance_schema.global_variables
--echo WHERE VARIABLE_NAME='$vector_build_threads_variable';
--disable_query_log
--eval SELECT * FROM performance_schema.global_variables WHERE VARIABLE_NAME='$vector_build_threads_variable'
--enable_query_log
--echo SELECT * FROM performance_schema.session_variables
--echo WHERE VARIABLE_NAME='$vector_build_threads_variable';
--disable_query_log
--eval SELECT * FROM performance_schema.session_variables WHERE VARIABLE_NAME='$vector_build_threads_variable'
--enable_query_log

--Error ER_INCORRECT_GLOBAL_LOCAL_VAR
--eval SELECT @@session.$vector_build_threads_variable

--Error ER_BAD_FIELD_ERROR
--eval SELECT $vector_build_threads_variable

--echo # Runtime changes are allowed.
--eval SET GLOBAL $vector_build_threads_variable=4
--eval SELECT @@global.$vector_build_threads_variable

--eval SET GLOBAL $vector_build_threads_variable=65535
--eval SELECT @@global.$vector_build_threads_variable

--Error ER_WRONG_VALUE_FOR_VAR
--eval SET GLOBAL $vector_build_threads_variable=65536

--Error ER_WRONG_VALUE_FOR_VAR
--eval SET GLOBAL $vector_build_threads_variable=-1

--eval SET GLOBAL $vector_build_threads_variable=DEFAULT
--eval SELECT @@global.$vector_build_threads_variable

--echo # End of tests.
