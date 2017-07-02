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

"""TensorBoard helper routine module.

This module is basically a dumpster for really generic succinct helper
routines that don't pull in any heavyweight dependencies aside from
TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import locale
import logging
import os
import sys
import threading

import tensorflow as tf


def setup_logging():
  """Configures Python logging the way the TensorBoard team likes it.

  This should be called at the beginning of a main function.
  """
  # TODO(jart): Make the default TensorFlow logger behavior great again.
  locale.setlocale(locale.LC_ALL, "")
  handler = LogHandler(sys.stderr)
  handler.setFormatter(LogFormatter())
  logging.getLogger('tensorflow').handlers = [handler]
  logging.currentframe = _hack_the_main_frame


def closeable(klass):
  """Makes a class with a close method able to be a context manager.

  This decorator is a great way to avoid having to choose between the
  boilerplate of __enter__ and __exit__ methods, versus the boilerplate
  of using contextlib.closing on every with statement.

  Args:
    klass: The class being decorated.

  Raises:
    ValueError: If class didn't have a close method, or already
        implements __enter__ or __exit__.
  """
  if 'close' not in klass.__dict__:
    # coffee is for closers
    raise ValueError('Class does not define a close() method: ' + klass)
  if '__enter__' in klass.__dict__ or '__exit__' in klass.__dict__:
    raise ValueError('Class already defines __enter__ or __exit__: ' + klass)
  klass.__enter__ = lambda self: self
  klass.__exit__ = lambda self, exc_type, exc_val, exc_tb: self.close()
  return klass


def close_all(resources):
  """Safely closes multiple resources.

  The close method on all resources is guaranteed to be called. If
  multiple close methods throw exceptions, then the least recent ones
  will be logged via tf.logging.error.

  Args:
    resources: An iterable of object instances whose classes implement
        the close method.

  Raises:
    Exception: To rethrow the last exception raised by a close method.
  """
  badness = None
  for resource in resources:
    try:
      resource.close()
    except Exception as e:
      if badness is not None:
        tf.logging.error('Suppressing close(%s) failure: %s', resource, e)
      badness = e
  if badness is not None:
    raise badness


def add_commas(n):
  """Adds locale specific thousands group separators.

  :type n: int
  :rtype: str
  """
  return locale.format('%d', n, grouping=True)


class LogFormatter(logging.Formatter):
  """Google style log formatter.

  The format is in essence the following:

      [DIWEF]mmdd hh:mm:ss.uuuuuu thread_name file:line] msg

  This class is meant to be used with LogHandler.
  """

  DATE_FORMAT = '%m%d %H:%M:%S'
  LOG_FORMAT = ('%(levelname)s%(asctime)s %(threadName)s '
                '%(filename)s:%(lineno)d] %(message)s')

  LEVEL_NAMES = {
      logging.FATAL: 'F',
      logging.ERROR: 'E',
      logging.WARN: 'W',
      logging.INFO: 'I',
      logging.DEBUG: 'D',
  }

  def __init__(self):
    """Creates new instance."""
    super(LogFormatter, self).__init__(LogFormatter.LOG_FORMAT,
                                       LogFormatter.DATE_FORMAT)

  def format(self, record):
    """Formats the log record.

    :type record: logging.LogRecord
    :rtype: str
    """
    record.levelname = LogFormatter.LEVEL_NAMES[record.levelno]
    return super(LogFormatter, self).format(record)

  def formatTime(self, record, datefmt=None):
    """Return creation time of the specified LogRecord as formatted text.

    :type record: logging.LogRecord
    :rtype: str
    """
    return (super(LogFormatter, self).formatTime(record, datefmt) +
            '.%06d' % (record.created % 1 * 1e6))


class Ansi(object):
  """ANSI terminal codes container."""

  BOLD = '\x1b[1m'
  CURSOR_HIDE = '\x1b[?25l'
  CURSOR_SHOW = '\x1b[?25h'
  FLIP = '\x1b[7m'
  MAGENTA = '\x1b[35m'
  RED = '\x1b[31m'
  RESET = '\x1b[0m'
  YELLOW = '\x1b[33m'


class LogHandler(logging.StreamHandler):
  """Log handler that supports ANSI colors and ephemeral records.

  Colors are applied on a line-by-line basis to non-INFO records. The
  goal is to help the user visually distinguish meaningful information,
  even when logging is verbose.

  Ephemeral log records, when emitted to a teletype emulator, only
  display on the final row, and get overwritten as soon as another
  ephemeral record is outputted. Ephemeral records are also sticky. If
  a normal record is written then the previous ephemeral record is
  restored right beneath it. When an ephemeral record with an empty
  message is emitted, then the last ephemeral record turns into a
  normal record and is allowed to spool.
  """

  EPHEMERAL = '.ephemeral'  # Name suffix for ephemeral loggers.

  COLORS = {
      logging.FATAL: Ansi.BOLD + Ansi.RED,
      logging.ERROR: Ansi.RED,
      logging.WARN: Ansi.YELLOW,
      logging.INFO: '',
      logging.DEBUG: Ansi.MAGENTA,
  }

  def __init__(self, stream):
    super(LogHandler, self).__init__(stream)
    self._stream = stream
    self._disable_flush = False
    self._is_tty = stream.isatty() and os.name != 'nt'
    self._ephemeral = u''

  def emit(self, record):
    """Emits a log record.

    :type record: logging.LogRecord
    """
    self.acquire()
    try:
      is_ephemeral = (self._is_tty and
                      record.name.endswith(LogHandler.EPHEMERAL))
      if self._is_tty:
        self._stream.write(LogHandler.COLORS[record.levelno])
      if is_ephemeral:
        ephemeral = record.getMessage()
        if ephemeral:
          self._clear_line()
          self._stream.write(ephemeral.encode('utf-8'))
        else:
          self._stream.write('\n')
        self._ephemeral = ephemeral
      else:
        self._clear_line()
        self._disable_flush = True  # prevent double flush
        super(LogHandler, self).emit(record)
        self._disable_flush = False
      if self._is_tty:
        self._stream.write(Ansi.RESET)
      if not is_ephemeral and self._ephemeral:
        self._stream.write(self._ephemeral.encode('utf-8'))
      self.flush()
    finally:
      self._disable_flush = False
      self.release()

  def flush(self):
    """Flushes output stream."""
    self.acquire()
    try:
      if not self._disable_flush:
        super(LogHandler, self).flush()
    finally:
      self.release()

  def _clear_line(self):
    if self._is_tty and self._ephemeral:
      # Our calculation of length won't be perfect due to ANSI codes,
      # but we can make it a little bit better by scrubbing these.
      text = self._ephemeral.replace('\x1b[', '')
      self._stream.write('\r' + ' ' * len(text) + '\r')


def _hack_the_main_frame():
  """Returns caller frame and skips over tf_logging.

  This works around a bug in TensorFlow's open source logging module
  where the Python logging module thinks are log entries are emitted by
  tf_logging.py delegate functions.
  """
  if hasattr(sys, '_getframe'):
    frame = sys._getframe(3)
  else:
    try:
      raise Exception
    except Exception:
      frame = sys.exc_info()[2].tb_frame.f_back
  if (frame is not None and
      hasattr(frame.f_back, "f_code") and
      'tf_logging.py' in frame.f_back.f_code.co_filename):
    return frame.f_back
  return frame
