let $have_vector_index = `SELECT COUNT(*) > 0 FROM performance_schema.global_variables WHERE variable_name REGEXP '^vector_'`;
if (!$have_vector_index)
{
  --skip Test requires 'have_vector_index'
}
