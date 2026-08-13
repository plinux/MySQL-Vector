--source include/have_vector_index.inc
--source include/have_debug.inc
--source include/not_windows.inc
--source include/have_binlog_format_row.inc

--let $MYSQLD_DATADIR= `select @@datadir;`
--let $assert_escape= 1
