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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import threading

import tensorflow as tf

from tensorboard import loader
from tensorboard import test_util
from tensorboard import util


class LoaderTestCase(tf.test.TestCase):
  def __init__(self, *args, **kwargs):
    super(LoaderTestCase, self).__init__(*args, **kwargs)
    self.clock = test_util.FakeClock()

  def _save_string(self, name, data):
    """Writes new file to temp directory.

    :type name: str
    :type data: str
    """
    path = os.path.join(self.get_temp_dir(), name)
    with open(path, 'wb') as writer:
      writer.write(tf.compat.as_bytes(data))
    return path

  def _save_records(self, name, records):
    """Writes new record file to temp directory.

    :type name: str
    :type records: list[str]
    :rtype: str
    """
    path = os.path.join(self.get_temp_dir(), name)
    with RecordWriter(path) as writer:
      for record in records:
        writer.write(record)
    return path


class BufferedRecordReaderTest(LoaderTestCase):
  def __init__(self, *args, **kwargs):
    super(BufferedRecordReaderTest, self).__init__(*args, **kwargs)
    self.BufferedRecordReader = functools.partial(
        loader.BufferedRecordReader, clock=self.clock)

  def testNoReads_closeWorks(self):
    path = self._save_records('empty.records', [])
    self.BufferedRecordReader(path).close()

  def testEmptyFile_returnsNoneRecords(self):
    path = self._save_records('empty.records', [])
    with self.BufferedRecordReader(path) as reader:
      self.assertIsNone(reader.get_next_record())
      self.assertIsNone(reader.get_next_record())

  def testGetSizeWhenNoReadsHappened_callsStatAnyway(self):
    path = self._save_string('empty.records', 'abc')
    with self.BufferedRecordReader(path) as reader:
      self.assertEqual(3, reader.get_size())

  def testGetNextRecord_worksWithGoodOffsets(self):
    path = self._save_records('foobar.records', ['foo', 'bar'])
    with self.BufferedRecordReader(path) as reader:
      record = reader.get_next_record()
      self.assertEqual('foo', record.record)
      self.assertLess(record.offset, reader.get_size())
      record = reader.get_next_record()
      self.assertEqual('bar', record.record)
      self.assertEqual(record.offset, reader.get_size())
      record = reader.get_next_record()
      self.assertIsNone(record)


class RecordReaderTest(LoaderTestCase):
  def testNoReads_closeWorks(self):
    path = self._save_records('empty.records', [])
    loader.RecordReader(path).close()

  def testEmptyFile_returnsNoneRecords(self):
    path = self._save_records('empty.records', [])
    with loader.RecordReader(path) as reader:
      self.assertIsNone(reader.get_next_record())
      self.assertIsNone(reader.get_next_record())

  def testGetNextRecord_worksWithGoodOffsets(self):
    path = self._save_records('foobar.records', ['foo', 'bar'])
    with loader.RecordReader(path) as reader:
      record = reader.get_next_record()
      self.assertEqual('foo', record.record)
      record = reader.get_next_record()
      self.assertEqual('bar', record.record)
      record = reader.get_next_record()
      self.assertIsNone(record)


@util.closeable
class RecordWriter(object):
  def __init__(self, path):
    self.path = tf.compat.as_bytes(path)
    self._writer = self._make_writer()

  def write(self, record):
    if not self._writer.WriteRecord(tf.compat.as_bytes(record)):
      raise IOError('Failed to write record to ' + self.path)

  def close(self):
    self._writer.Close()

  def _make_writer(self):
    with tf.errors.raise_exception_on_not_ok_status() as status:
      return tf.pywrap_tensorflow.PyRecordWriter_New(self.path, '', status)


if __name__ == '__main__':
  tf.test.main()
