# Validate a vector sysvar that is covered by a grouped sys_vars test.
--source include/have_vector_index.inc

if (`SELECT '$vector_sysvar_name' = ''`)
{
  --die vector_sysvar_alias_basic.inc requires vector_sysvar_name
}

--echo #
--echo # Variable name : $vector_sysvar_name
--echo # Covered by grouped vector sys_vars tests.
--echo #

--disable_query_log
--disable_result_log
--let $global_count= `SELECT COUNT(*) FROM performance_schema.global_variables WHERE VARIABLE_NAME = '$vector_sysvar_name'`
if (!$global_count)
{
  --die $vector_sysvar_name is not registered as a global variable
}
--let $session_count= `SELECT COUNT(*) FROM performance_schema.session_variables WHERE VARIABLE_NAME = '$vector_sysvar_name'`
if (!$session_count)
{
  --die $vector_sysvar_name is not visible through session variables
}
--enable_result_log
--enable_query_log
