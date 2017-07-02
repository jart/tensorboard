# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TensorBoard data ingestion module.

This module is for loading the log directories created by TensorFlow's
FileWriter into a SQL database. We do this transactionally. In a single
thread, with a single database connection, we read a single tf.Event
from each of the runs being loaded. We hand off each event to each
plugin, along with the database connection. The plugins do their thing,
and then the transaction gets committed.

If the user is passing step counts to the add_summary, then this module
is able to skip reading restarted event logs entirely. However, this
module also allows event logs to be read as they're being actively
written by TensorFlow, so plugins should not assume they'll receive
event steps in monotonic order.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import itertools
import logging
import os
import re
import sys
import time
import threading
import uuid

import six
import tensorflow as tf

from tensorboard import schema
from tensorboard import util

_EVENT_LOG_PATH_PATTERN = re.compile(
    r'\.tfevents\.(?P<timestamp>\d+).(?P<hostname>[-.0-9A-Za-z]+)$')
_SHORTEN_EVENT_LOG_PATH_PATTERN = re.compile(r'(?:[^/\\]+[/\\])?(?:[^/\\]+)$')


class Record(collections.namedtuple('Record', ('record', 'offset'))):
  """Value class for a record returned by RecordReader.

  Fields:
    record: The byte string record that was read.
    offset: The byte offset in the file *after* this record was read.

  :type record: str
  :type offset: int
  """
  __slots__ = ()  # Enforces use of only tuple fields.


@util.closeable
@six.python_2_unicode_compatible
class FileLoader(object):
  """Utility for loading many runs.

  Fields:
    is_tapped: Whether or not the EOF for each event log has been
        observed for all runs and experiments.
  """

  MINIMUM_BATCH_SIZE = 23

  def __init__(self, db_conn):
    """Creates new instance.

    Args:
      db_conn: A PEP 249 Connection object.
    """
    self.is_tapped = True
    self._db_conn = db_conn
    self._runs = set()  # type: set[RunLoader]
    self._logdirs = {}  # type: dict[str, dict[str, RunLoader]]

  def add_log_directory(self, logdir):
    """Synchronizes log directory with loader state.

    This method can add a single run, or multiple runs, to the loader
    state and infer their names from the directory structure. Their
    directories will also be monitored automatically to detect new
    event logs. It will also delete runs from the database when their
    event logs are all deleted from the file system.

    Args:
      logdir: A string path to a run directory containing event logs,
          or an experiment directory whose sub-directories contain
          event logs.

    Raises:
      ValueError: If logdir isn't a directory.

    :type logdir: str
    """
    if not tf.gfile.IsDirectory(logdir):
      raise ValueError('Not a directory: ' + logdir)
    if logdir not in self._logdirs:
      self._logdirs[logdir] = {}
    logdir_runs = self._logdirs[logdir]
    ordered_event_logs = get_event_logs(logdir)
    tf.logging.debug('Found %d event logs in %s',
                     len(ordered_event_logs), logdir)
    for run_dir, logs in itertools.groupby(ordered_event_logs,
                                           lambda l: os.path.dirname(l.path)):
      run_dir = os.path.normcase(run_dir)
      if run_dir in logdir_runs:
        run = logdir_runs[run_dir]
      else:
        run_name = os.path.relpath(run_dir, logdir)
        if run_name == '.':
          run_name = _get_basename(logdir)
        run = self.add_run(run_name)
        logdir_runs[run_dir] = run
      for log in logs:
        run.add_event_log(log)
    event_logs = set(ordered_event_logs)
    for run_dir, run in logdir_runs.items():
      if not all(log in event_logs for log in run.get_event_logs()):
        tf.logging.warning('Deleting %s', run)
        self.delete_run(run)
        del logdir_runs[run_dir]
    if not logdir_runs:
      del self._logdirs[logdir]

  def add_run(self, name):
    """Adds run to loader state.

    If the run does not exist in the database, it will be created.

    Args:
      name: The display name of the run, which is a unique index field
          in the database.

    Returns:
      A RunLoader instance with no EventLog objects; those must be
      added by the caller.

    :type name: str
    :rtype: RunLoader
    """
    name = tf.compat.as_text(name)
    with contextlib.closing(self._db_conn.cursor()) as c:
      c.execute('SELECT id FROM runs where name = ?', (name,))
      row = c.fetchone()
      if row:
        tf.logging.info('Updating run: %s', name)
        run_id = row[0]
      else:
        tf.logging.info('Added run: %s', name)
        run_id = str(uuid.uuid4())
        c.execute('INSERT INTO runs (id, name) VALUES (?, ?)', (run_id, name))
    run = RunLoader(self._db_conn, run_id, name)
    self._runs.add(run)
    self.is_tapped = False
    return run

  def delete_run(self, run):
    """Removes run from loader state and deletes it from the database.

    Please note that this only deletes the entry from the `runs` table.
    The contents of other tables with a foreign key on this run will be
    left intact and should be removed periodically by a cron job.

    :type run: RunLoader
    """
    if run not in self._runs:
      raise ValueError('%s not loaded in %s' % (run, self))
    tf.logging.warning('Deleting run: %s', run.name)
    with contextlib.closing(self._db_conn.cursor()) as c:
      c.execute('DELETE FROM runs where id = ?', (run.id,))
    run.close()
    self._runs.remove(run)

  def get_some_events(self):
    """Returns a batch of events from various runs or empty if tapped.

    How many events this method returns is undefined. However, at the
    moment, this method tries to read one event from all runs being
    loaded, so a large number of runs will all load gradually together.

    :rtype: list[tuple[RunLoader, tf.Event]]
    """
    result = []
    while len(result) < FileLoader.MINIMUM_BATCH_SIZE:
      moar = self._collect_events()
      if not moar:
        break
      result.extend(moar)
    if not result:
      self._scan_for_new_and_deleted_runs()
      result.extend(self._collect_events(force=True))
    self.is_tapped = not result
    return result

  def get_offset(self):
    """Returns number of bytes read across all event log files.

    :rtype: int
    """
    return sum(el.get_offset() for el in self._runs)

  def get_size(self):
    """Returns sum of byte lengths of event log files.

    :rtype: int
    """
    return sum(el.get_size() for el in self._runs)

  def close(self):
    """Closes all event log readers.

    This method may be called multiple times. Further read operations
    are only permitted if new runs or log directories are added, with
    timestamps greater than those previously observed.

    Raises:
      Exception: To propagate the most recent exception thrown by the
          EventLog close method. Suppressed exceptions are logged.
    """
    util.close_all(self._runs)

  def _collect_events(self, force=False):
    result = []
    for run in self._runs:
      if run.is_tapped and not force:
        continue
      event = run.get_next_event()
      if event is not None:
        result.append((run, event))
    return result

  def _scan_for_new_and_deleted_runs(self):
    for directory in self._logdirs.keys():
      self.add_log_directory(directory)

  def __str__(self):
    return u'Loader{progress=%d/%d, runs=%s}' % (
        self.get_offset(), self.get_size(), [r.name for r in self._runs])


@util.closeable
@six.python_2_unicode_compatible
class RunLoader(object):
  """Utility for loading events from a directory of event logs.

  Fields:
    run_id: Database primary key of this run.
    run_name: Display name of this run.
    is_tapped: Whether or not the EOF for each event log has been
        observed.
  """

  def __init__(self, db_conn, id, name):
    """Creates new instance.

    Args:
      db_conn: A PEP 249 Connection object.
      id: Primary key of run in `runs` table in DB, which should
          already be inserted.
      name: Display name of run.

    :type id: str
    :type name: str
    """
    self._db_conn = db_conn
    self.id = tf.compat.as_text(id)
    self.name = tf.compat.as_text(name)
    self.is_tapped = True
    self._logs = []  # type: list[EventLog]
    self._i = 0
    self._entombed_progress = 0
    self._has_new_stuff = False

  def get_event_logs(self):
    """Returns all event logs being considered.

    This might not be all the event logs that have been added, because
    sometimes they get optimized away.

    Yields:
      An EventLog instance.

    :rtype: collections.Iterable[EventLog]
    """
    for log in self._logs:
      yield log

  def add_event_log(self, log):
    """Adds event log to run loader.

    Event logs must be added monotonically, based on the timestamp in
    the filename.

    Args:
      log: An EventLog instance.

    Returns:
      True if log was actually added.

    :type log: EventLog
    :rtype: bool
    """
    if self._logs and log <= self._logs[-1]:
      return False
    with contextlib.closing(self._db_conn.cursor()) as c:
      c.execute('SELECT offset FROM event_logs WHERE run_id = ? AND path = ?',
                (self.id, log.path))
      row = c.fetchone()
      if row:
        log.set_offset(row[0])
      else:
        c.execute(
            'INSERT INTO event_logs (run_id, path, offset) VALUES (?, ?, ?)',
            (self.id, log.path, 0))
    self._logs.append(log)
    self.is_tapped = False
    self._has_new_stuff = True
    tf.logging.debug('Adding %s', log)
    return True

  def get_next_event(self):
    """Returns next tf.Event from the event logs or None if tapped.

    :rtype: tf.Event
    """
    if self._i == len(self._logs):
      return None
    if self._has_new_stuff:
      self._fast_forward_over_things_we_have_read()
      self._fast_forward_over_restarted_runs()
      self._cleanup()
      self._has_new_stuff = False
    event = None
    while True:
      log = self._logs[self._i]
      event = log.get_next_event()
      if event is not None:
        with contextlib.closing(self._db_conn.cursor()) as c:
          c.execute(
              'UPDATE event_logs SET offset = ? WHERE run_id = ? AND path = ?',
              (log.get_offset(), self.id, log.path))
        break
      if self._i == len(self._logs) - 1:
        self.is_tapped = True
        break
      self._i += 1
      self._cleanup()
    return event

  def get_offset(self):
    """Returns number of bytes read across all event log files.

    :rtype: int
    """
    return sum(el.get_offset() for el in self._logs) + self._entombed_progress

  def get_size(self):
    """Returns sum of byte lengths of event log files.

    :rtype: int
    """
    return sum(el.get_size() for el in self._logs) + self._entombed_progress

  def close(self):
    """Closes all event log readers.

    This method may be called multiple times. Further read operations
    are only permitted if new event logs are added, with timestamps
    greater than those previously observed.

    Raises:
      Exception: To propagate the most recent exception thrown by the
          EventLog close method. Suppressed exceptions are logged.
    """
    self._i = len(self._logs)
    util.close_all(self._logs)

  def _fast_forward_over_things_we_have_read(self):
    for i in range(len(self._logs) - 1, self._i + 1, -1):
      if self._logs[i].get_offset():
        self._i = i
        break

  def _fast_forward_over_restarted_runs(self):
    skip_to = self._i
    for i in range(len(self._logs) - 2, self._i, -1):
      a = self._logs[i].get_first_step()
      b = self._logs[i + 1].get_first_step()
      if b is not None and (a is None or b <= a):
        skip_to = i + 1
        break
    for i in range(self._i, skip_to):
      tf.logging.warning('Skipping %s because %s reset the step counter',
                         self._logs[i], self._logs[skip_to])
    self._i = skip_to

  def _cleanup(self):
    # Last event log has to be preserved so we can continue enforcing
    # monotonicity. We entomb offset because that also has to be
    # monotonic, but the size does not.
    if 0 < self._i < len(self._logs):
      deleted = self._logs[:self._i]
      self._logs = self._logs[self._i:]
      self._i = 0
      self._entombed_progress += sum(l.get_offset() for l in deleted)
      util.close_all(deleted)

  def __str__(self):
    offset = self.get_offset()
    if offset:
      return u'RunLoader{name=%s, offset=%d}' % (self.name, offset)
    else:
      return u'RunLoader{%s}' % self.path


@util.closeable
@six.python_2_unicode_compatible
class RecordReader(object):
  """Pythonic veneer around PyRecordReader."""

  def __init__(self, path, start_offset=0):
    """Creates new instance.

    Args:
      path: Path of file. This can be on a remote file system if the
          TensorFlow build supports it.
      start_offset: Byte offset to seek in file once it's opened.

    :type path: str
    :type start_offset: int
    """
    self.path = tf.compat.as_text(path)
    self._start_offset = start_offset
    self._reader = None  # type: tf.pywrap_tensorflow.PyRecordReader
    self._is_closed = False

  def get_next_record(self):
    """Reads record from file.

    Returns:
      A Record or None if no more were available.

    Raises:
      IOError: On open or read error, or if close was called.

    :type reader: tf.PyRecordReader
    :rtype: Record
    """
    if self._is_closed:
      raise IOError('%s is closed' % self)
    if self._reader is None:
      self._reader = self._open()
    try:
      with tf.errors.raise_exception_on_not_ok_status() as status:
        self._reader.GetNext(status)
    except (tf.errors.DataLossError, tf.errors.OutOfRangeError):
      # We ignore partial read exceptions, because a record may be truncated.
      # PyRecordReader holds the offset prior to the failed read, so retrying
      # will succeed.
      return None
    return Record(self._reader.record(), self._reader.offset())

  def get_offset(self):
    """Returns current offset in file.

    Raises:
      IOError: If close was called.

    :rtype: int
    """
    if self._is_closed:
      raise IOError('%s is closed' % self)
    if self._reader is None:
      return self._start_offset
    return self._reader.offset()

  def get_size(self):
    """Returns byte length of file.

    This method can be called after the instance has been closed.

    :rtype: int
    """
    # TODO(jart): Size check here.
    return tf.gfile.Stat(self.path).length

  def close(self):
    """Closes record reader if open.

    Further reads are not permitted after this method is called.
    """
    if self._is_closed:
      return
    if self._reader is not None:
      self._reader.Close()
    self._is_closed = True
    self._reader = None

  def _open(self):
    with tf.errors.raise_exception_on_not_ok_status() as status:
      return tf.pywrap_tensorflow.PyRecordReader_New(
          tf.resource_loader.readahead_file_path(tf.compat.as_bytes(self.path)),
          self._start_offset, '', status)

  def __str__(self):
    return u'RecordReader{%s}' % self.path


@util.closeable
@six.python_2_unicode_compatible
class BufferedRecordReader(object):
  """Wrapper around RecordReader that does threaded read-ahead.

  This class implements the same interface as RecordReader. It prevents
  remote file systems from devastating loader performance. It does not
  degrade throughput on local file systems.

  The thread is spawned when the first read operation happens. The
  thread will diligently try to buffer records in the background. Its
  goal is to sleep as much as possible without blocking read operations.

  This class is thread safe. It can be used from multiple threads
  without any need for external synchronization.
  """

  READ_AHEAD_AGGRESSION = 2.3
  READ_AHEAD_BYTES = 64 * 1024 * 1024
  STAT_INTERVAL_SECONDS = 4.0

  def __init__(self, path,
               start_offset=0,
               read_ahead=None,
               clock=time.time):
    """Creates new instance.

    The i/o thread is not started until the first read happens.

    Args:
      path: Path of file. This can be on a remote file system if the
          TensorFlow build supports it.
      start_offset: Byte offset to seek in file once it's opened.
      read_ahead: The number of record bytes to buffer into memory
          before the thread starts blocking. This value must be >0 and
          the default is BufferedRecordReader.DEFAULT_READ_AHEAD.
      clock: Function returning a float with the number of seconds
          since the UNIX epoch in zulu time.

    :type path: str
    :type start_offset: int
    :type read_ahead: int
    :type clock: () -> float
    """
    self.path = tf.compat.as_text(path)
    self._read_ahead = read_ahead or BufferedRecordReader.READ_AHEAD_BYTES
    self._clock = clock
    self._is_started = False
    self._is_closed = False
    self._is_tapped = False
    self._size = -1
    self._last_stat = 0.0
    self._buffered = 0
    self._reader = RecordReader(self.path, start_offset)
    self._records = collections.deque()  # type: collections.deque[Record]
    self._read_exception = None  # type: Exception
    self._close_exception = None  # type: Exception
    self._lock = threading.Lock()
    self._please_sir_i_want_some_more = threading.Condition(self._lock)
    self._no_soup_for_you = threading.Condition(self._lock)
    self._thread = threading.Thread(target=self._run,
                                    name=_shorten_event_log_path(self.path))

  def get_size(self):
    """Returns byte length of file.

    In the average case, this method will not block. However, if the
    i/o thread has not yet computed this value, then this method will
    block on a stat call.

    This method can be called after the instance has been closed.

    Returns:
      The byte length of file, which might increase over time, but is
      guaranteed to never decrease. It's also guaranteed that it will
      be greater than or equal to the offset field of any Record.

    Raises:
      IOError: If the stat call failed.

    :rtype: int
    """
    with self._lock:
      size = self._size
    if size == -1:
      size = self._reader.get_size()
      with self._lock:
        # TODO(jart): Size check here.
        if size > self._size:
          self._size = size
    return size

  def get_next_record(self):
    """Reads one record.

    When this method is first called, it will spawn the thread and
    block until a record is read. Once the thread starts, it will queue
    up records which can be read without blocking. The exception is
    when we reach the end of the file, in which case each repeated call
    will be synchronous. There is no background polling. If new data is
    appended to the file, new records won't be buffered until this
    method is invoked again. The caller should take care to meter calls
    to this method once it reaches the end of file, lest they impact
    performance.

    Returns:
      A Record object, or None if there are no more records available
      at the moment.

    Raises:
      IOError: If this instance has been closed.
      Exception: To propagate any exceptions that may have been thrown
          by the read operation in the other thread. If an exception is
          thrown, then all subsequent calls to this method will rethrow
          that same exception.

    :rtype: Record
    """
    with self._lock:
      if self._is_closed:
        raise IOError('%s is closed' % self)
      if not self._is_started:
        self._thread.start()
        self._is_started = True
      else:
        record = self._get_record()
        if record is not None:
          return record
        self._is_tapped = False
        self._please_sir_i_want_some_more.notify()
      while (self._read_exception is None and
             not self._is_tapped and
             not self._records):
        self._no_soup_for_you.wait()
      return self._get_record()

  def close(self):
    """Closes event log reader if open.

    If the i/o thread is running, this method blocks until it has been
    shut down.

    Further reads are not permitted after this method is called.

    Raises:
      Exception: To propagate any exceptions that may have been thrown
          by the close operation in the other thread. If an exception
          is thrown, then all subsequent calls to this method will
          rethrow that same exception.
    """
    with self._lock:
      if not self._is_closed:
        self._is_closed = True
        if not self._is_started:
          return
        self._please_sir_i_want_some_more.notify()
      while self._reader is not None:
        self._no_soup_for_you.wait()
      if self._close_exception is not None:
        raise self._close_exception

  def _get_record(self):
    if self._read_exception is not None:
      raise self._read_exception
    if not self._records:
      return None
    record = self._records.popleft()
    self._buffered -= len(record.record)
    if self._should_wakeup():
      self._please_sir_i_want_some_more.notify()
    return record

  def _run(self):
    while True:
      if self._should_stat():
        self._stat()
      with self._lock:
        while not self._should_wakeup():
          self._please_sir_i_want_some_more.wait()
        if self._is_closed:
          try:
            self._reader.close()
            tf.logging.debug('Closed')
          except Exception as e:
            self._close_exception = e
            tf.logging.debug('Close failed: %s', e)
          self._reader = None
          self._no_soup_for_you.notify_all()
          return
        if not self._should_rebuffer():
          continue
        # Calculate a good amount of data to read outside the lock.
        # The less we have buffered, the less re-buffering we'll do.
        # We want to minimize wait time in the other thread.
        x = float(self._buffered)
        c = float(self._read_ahead)
        s = BufferedRecordReader.READ_AHEAD_AGGRESSION
        want = int(min(c - x, s/c * x**s + 1))
      tf.logging.debug('Waking up to read %s bytes', util.add_commas(want))
      records = []
      read_exception = None
      try:
        while want > 0:
          record = self._reader.get_next_record()
          if record is None:
            break
          records.append(record)
          want -= len(record.record)
      except Exception as e:
        read_exception = e
      with self._lock:
        if read_exception is not None:
          self._read_exception = read_exception
          self._is_closed = True
        elif not records:
          self._is_tapped = True
        else:
          for record in records:
            self._records.append(record)
            self._buffered += len(record.record)
        self._no_soup_for_you.notify_all()

  def _should_wakeup(self):
    return (self._is_closed or
            self._should_rebuffer() or
            self._should_stat())

  def _should_rebuffer(self):
    return (not self._is_tapped and
            (float(self._buffered) <
             self._read_ahead / BufferedRecordReader.READ_AHEAD_AGGRESSION))

  def _should_stat(self):
    return (self._reader.get_offset() > self._size or
            (self._last_stat <=
             self._clock() - BufferedRecordReader.STAT_INTERVAL_SECONDS))

  def _stat(self):
    now = self._clock()
    size = self._reader.get_size()
    minimum = max(self._size, self._reader.get_offset())
    if size < minimum:
      tf.logging.warning('File shrunk while reading: %s', self.path)
      size = minimum
    with self._lock:
      self._size = size
      self._last_stat = now

  def __str__(self):
    return u'BufferedRecordReader{%s}' % self.path


@util.closeable
@functools.total_ordering
@six.python_2_unicode_compatible
class EventLog(object):
  """Helper class for reading from event log files.

  This class is wrapper around BufferedRecordReader that operates on
  record files containing tf.Event protocol buffers.

  Instances of this class can be used in hash tables and binary trees.
  """

  FIRST_STEP_PEEKS = 5

  def __init__(self, path, record_reader_factory=BufferedRecordReader):
    """Creates new instance.

    Args:
      path: Path of event log file.
      record_reader_factory: A reference to the constructor of a class
          that implements the same interface as RecordReader.

    :type path: str
    :type record_reader_factory: (str, int) -> RecordReader
    """
    a = record_reader_factory('', 3)
    self.path = tf.compat.as_text(path)
    m = _EVENT_LOG_PATH_PATTERN.search(self.path)
    if not m:
      raise ValueError('Bad event log path: ' + self.path)
    self.timestamp = int(m.group('timestamp'))
    self.hostname = m.group('hostname')
    self._offset = 0
    self._record_reader_factory = record_reader_factory
    self._reader = record_reader_factory(self.path)
    self._key = (os.path.dirname(self.path), self.timestamp)
    self._first_step = None  # type: int
    self._events = False
    self._observed_zero_step = False
    self._update_first_step_invocations = 0

  def get_next_event(self):
    """Reads an event proto from the file.

    Returns:
      A tf.Event or None if no more records exist in the file. Please
      note that the file remains open for subsequent reads in case more
      are appended later.

    :rtype: tf.Event
    """
    record = self._reader.get_next_record()
    if record is None:
      return None
    self._offset = record.offset
    event = tf.Event()
    event.ParseFromString(record.record)
    self._update_first_step(event)
    return event

  def get_offset(self):
    """Returns current byte offset in file.

    This does not take read-ahead into consideration. It returns the
    offset in the file after the record storing the last event returned
    by get_next_event.

    :rtype: int
    """
    return self._offset

  def set_offset(self, offset):
    """Seeks in file to byte offset.

    :rtype: int
    """
    if offset != self._offset:
      self._reader.close()
      self._reader = self._record_reader_factory(self.path, offset)
      self._offset = offset

  def get_size(self):
    """Returns byte length of file.

    :rtype: int
    """
    return self._reader.get_size()

  def get_first_step(self):
    """Determines step count of first summary event.

    This method requires an open/read/close but memoizes the result. If
    the answer was already inferred from previous calls to
    get_next_event, then no i/o should be necessary.

    Returns:
      The step count of the first summary event, or None if there
      didn't appear to be any summary events, or the user isn't logging
      step counts.

    :rtype: int
    """
    if self._first_step is not None:
      return self._first_step
    if self._update_first_step_invocations >= EventLog.FIRST_STEP_PEEKS:
      return None
    with RecordReader(self.path) as reader:
      for _ in range(EventLog.FIRST_STEP_PEEKS):
        record = reader.get_next_record()
        if record is None:
          break
        event = tf.Event()
        event.ParseFromString(record.record)
        self._update_first_step(event)
    return self._first_step

  def close(self):
    """Closes event log reader if open.

    Further i/o is not permitted after this method is called.
    """
    self._reader.close()

  def _update_first_step(self, event):
    self._update_first_step_invocations += 1
    # proto3 doesn't let us distinguish between absent and 0.
    if event.step:
      step = 0 if self._observed_zero_step else event.step
      if self._first_step is None or step < self._first_step:
        self._first_step = step
    elif event.summary:
      self._observed_zero_step = True

  def __hash__(self):
    return hash(self._key)

  def __eq__(self, other):
    return self._key == other._key

  def __lt__(self, other):
    return self._key < other._key

  def __str__(self):
    offset = self.get_offset()
    if offset:
      return u'EventLog{path=%s, offset=%d}' % (self.path, offset)
    else:
      return u'EventLog{%s}' % self.path


@util.closeable
class Progress(object):
  """Terminal UI for displaying job progress in terms of bytes.

  On teletypes, this class will display a nice ephemeral unicode
  progress bar. Otherwise it just emits periodic log messages the old
  fashioned way.

  This class keeps track of the rate at which input is processed, as
  well as the rate it grows. These values are represented to the user
  using the DELTA and NAMBLA symbols.

  An alarm is displayed if the consumption rate falls behind the
  production rate. In order for this to be calculated properly, the
  sleep method of this class should be used rather than time.sleep.
  """

  BAR_INTERVAL_SECONDS = 0.25
  BAR_LOGGER = logging.getLogger('tensorflow' + util.LogHandler.EPHEMERAL)
  BAR_WIDTH = 45
  BLOCK_DARK = u'\u2593'
  BLOCK_LIGHT = u'\u2591'
  DELTA = u'\u2206'
  LOG_INTERVAL_SECONDS = 2.5
  NABLA = u'\u2207'
  RATE_WINDOW = 20.0

  def __init__(self, clock=time.time, sleep=time.sleep):
    """Creates new instance.

    Args:
      stream: File object to which bytes are written. If this is a
          teletype then a cool progress bar will be shown.
      clock: Function returning a float with the number of seconds
          since the UNIX epoch in zulu time.
      sleep: Injected time.sleep function.

    :type clock: () -> float
    :type sleep: (float) -> None
    """
    self._clock = clock
    self._sleep = sleep
    self._is_teletype = sys.stderr.isatty() and os.name != 'nt'
    self._offset = 0
    self._size = 0
    self._last_time = 0.0
    self._interval = (Progress.BAR_INTERVAL_SECONDS
                      if self._is_teletype else
                      Progress.LOG_INTERVAL_SECONDS)
    self._rate_offset = RateCounter(Progress.RATE_WINDOW)
    self._rate_size = RateCounter(Progress.RATE_WINDOW)

  def set_progress(self, offset, size):
    """Updates the progress bar state.

    This method will cause progress information to be occasionally
    written out.

    Args:
      offset: The number of bytes processed so far.
      size: The total number of bytes. This is allowed to increase or
          decrease, but it must remain greater than offset.

    Raises:
      ValueError: If offset is greater than size, or offset or size
          decreased from the last invocation.

    :type offset: int
    :type size: int
    """
    if offset > size:
      raise ValueError('offset (%d) can not exceed size (%d)' % (offset, size))
    now = self._clock()
    self._rate_offset.set_value(offset)
    self._rate_size.set_value(size)
    self._offset = offset
    self._size = size
    elapsed = now - self._last_time
    if elapsed < self._interval:
      return
    self._last_time = now
    self._show()

  def close(self):
    """Forces progress to be written to log.

    This method exists because we don't want the progress bar to say
    something like 98% once the file is done loading.
    """
    self._show()
    if self._is_teletype:
      # Instructs util.LogHandler to clear the ephemeral logging state.
      Progress.BAR_LOGGER.info('')

  def sleep(self, seconds):
    """Sleeps for a given number of seconds.

    Time spent sleeping in this method does not have a detrimental
    impact on the consumption rate.

    :type seconds: float
    """
    self._sleep(seconds)
    self._rate_offset.bump()

  def _show(self):
    if self._is_teletype:
      self._show_bar()
    else:
      self._show_log()

  def _show_log(self):
    tf.logging.info('Loaded %s', self._get_message())

  def _show_bar(self):
    sofar = int(float(self._offset) / self._size * Progress.BAR_WIDTH)
    bar = (Progress.BLOCK_DARK * sofar +
           Progress.BLOCK_LIGHT * (Progress.BAR_WIDTH - sofar))
    Progress.BAR_LOGGER.info(u'%s %s ' % (bar, self._get_message()))

  def _get_message(self):
    rate_offset = self._rate_offset.get_rate()  # summary processing speed
    rate_size = self._rate_size.get_rate()  # summary production speed
    message = u'%d%% of %s%s%s' % (
        int(float(self._offset) / self._size * 100.0),
        util.add_commas(self._size),
        self._get_rate_suffix(Progress.DELTA, rate_offset),
        self._get_rate_suffix(Progress.NABLA, rate_size))
    if rate_offset and rate_size and rate_offset < rate_size:
      # If TensorFlow is writing summaries to disk faster than we can
      # insert them into the database, that's kind of problematic.
      message += u' ' + self._make_red(u'[meltdown]')
    return message

  def _get_rate_suffix(self, symbol, rate):
    if not rate:
      return u''
    return u' %s %sB/s' % (symbol, util.add_commas(rate))

  def _make_red(self, text):
    if self._is_teletype:
      return (util.Ansi.BOLD +
              util.Ansi.RED +
              (util.Ansi.FLIP if self._offset % 2 == 0 else u'') +
              text +
              util.Ansi.RESET)
    return text


class RateCounter(object):
  """Utility class for tracking how much a number increases each second.

  The rate is calculated by averaging of samples within a time window,
  which weights recent samples more strongly.
  """

  def __init__(self, window, clock=time.time):
    """Creates new instance.

    Args:
      window: The maximum number of seconds across which rate is
          averaged. In practice, the rate might be averaged over a time
          period greater than window if set_value is being called less
          frequently than window.
      clock: Function returning a float with the number of seconds
          since the UNIX epoch in zulu time.

    :type window: float
    :type clock: () -> float
    """
    self._window = window
    self._clock = clock
    self._points = collections.deque()
    self._last_value = None  # type: int
    self._last_time = None  # type: float

  def get_rate(self):
    """Determines rate of increase in value per second averaged over window.

    Returns:
      An integer representing the rate or None if not enough
      information has been collected yet.

    :rtype: int
    """
    toto = 0.0
    points = []
    for i, (rate, weight, _) in enumerate(self._points):
      weight *= 1.0 / (i + 1)
      toto += weight
      points.append((rate, weight))
    return int(sum(w / toto * r for r, w in points))

  def set_value(self, value):
    """Sets number state.

    This method adds a delta between value and the value of the last
    time this method was called. Therefore the first invocation does
    not add a delta.

    Raises:
      ValueError: If value is less than the last value.

    :type value: int
    """
    now = self._clock()
    if self._last_value is None:
      self._last_value = value
      self._last_time = now
      return
    if value < self._last_value:
      raise ValueError('%d < %d' % (value, self._last_value))
    delta = value - self._last_value
    elapsed = now - self._last_time
    self._points.appendleft((delta / elapsed, elapsed, now))
    self._last_time = now
    self._last_value = value
    self._remove_old_points()

  def bump(self):
    """Makes time since last set_value count for nothing."""
    self._last_time = self._clock()

  def _remove_old_points(self):
    threshold = self._clock() - self._window
    while self._points:
      r, e, t = self._points.pop()
      if t > threshold:
        self._points.append((r, e, t))
        break


def is_event_log_file(path):
  """Returns True if path appears to be an event log file.

  :type path: str
  :rtype: bool
  """
  return bool(_EVENT_LOG_PATH_PATTERN.search(path))


def get_event_logs(directory):
  """Finds protobuf event log files.

  Args:
    directory: Path of directory.

  Returns:
    List of EventLog objects, ordered by directory name and timestamp.

  :type directory: str
  :rtype: list[EventLog]
  """
  logs = []
  for dirname, _, filenames in tf.gfile.Walk(directory):
    for filename in filenames:
      if is_event_log_file(filename):
        logs.append(EventLog(os.path.join(dirname, filename)))
  logs.sort()
  return logs


def _shorten_event_log_path(path):
  """Makes an event log path more human readable.

  Returns:
    Path containing only basename and the first parent directory name,
    if there is one.

  :type path: str
  :rtype: str
  """
  m = _SHORTEN_EVENT_LOG_PATH_PATTERN.search(path)
  if m:
    return m.group(0)
  return path


def _get_basename(path):
  """Gets base name of path.

  This is the same as os.path.basename, however it may potentially do
  i/o to handle a few edge cases, which would otherwise cause the
  result to be less meaningful, e.g. "." and "..".

  :type path: str
  :rtype: str
  """
  result = os.path.basename(os.path.normpath(path))
  if result in ('', '.', '..'):
    result = os.path.basename(os.path.realpath(path))
  return result


def main(argv=None):
  util.setup_logging()
  tf.logging.set_verbosity(tf.logging.DEBUG)

  # with EventLog(sys.argv[1]) as log:
  #   with Progress() as progress:
  #     while True:
  #       event = log.get_next_event()
  #       progress.set_progress(log.get_offset(), log.get_size())
  #       if event is None:
  #         break
  # sys.exit(0)

  import contextlib
  import sqlite3

  with sqlite3.connect('doodle.db') as db_conn:
    schema.setup_database(db_conn)
    try:
      with Progress() as progress, FileLoader(db_conn) as loader:
        loader.add_log_directory(argv[1])
        while True:
          events = loader.get_some_events()
          progress.set_progress(loader.get_offset(), loader.get_size())
          db_conn.commit()
          if not events:
            progress.sleep(5.0)
    except KeyboardInterrupt:
      db_conn.rollback()


if __name__ == '__main__':
  tf.app.run()
