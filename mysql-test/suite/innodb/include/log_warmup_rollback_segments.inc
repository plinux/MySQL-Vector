# Some redo log boundary tests require a small foreground transaction to fit
# inside one 512-byte log block. With many active rollback segments, the first
# transaction using a segment may initialize undo state and emit extra redo.
# Warm the configured segments before the boundary-sensitive part of a test so
# it can run with the server default instead of overriding innodb_rollback_segments.

--let $rollback_segments_warmup_count = `SELECT @@GLOBAL.innodb_rollback_segments`
--let $rollback_segments_warmup = 0

SET AUTOCOMMIT = 0;
while ($rollback_segments_warmup < $rollback_segments_warmup_count) {
  eval UPDATE t SET a = 99 WHERE a = $rollback_segments_warmup_value;
  ROLLBACK;
  --inc $rollback_segments_warmup
}
SET AUTOCOMMIT = 1;
