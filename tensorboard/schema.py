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

"""TensorBoard database schema module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib


def setup_database(db_conn):
  """Creates core tables needed by TensorBoard."""
  create_runs_table(db_conn)
  create_event_logs_table(db_conn)
  db_conn.commit()


def create_runs_table(db_conn):
  """Creates the runs table."""
  with contextlib.closing(db_conn.cursor()) as c:
    c.execute('''\
      CREATE TABLE IF NOT EXISTS runs (
        id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(255) NOT NULL
      )
    ''')
    c.execute('''\
      CREATE UNIQUE INDEX IF NOT EXISTS runs_index
      ON runs (name)
    ''')


def create_event_logs_table(db_conn):
  """Creates the runs table."""
  with contextlib.closing(db_conn.cursor()) as c:
    c.execute('''\
      CREATE TABLE IF NOT EXISTS event_logs (
        run_id VARCHAR(64) NOT NULL,
        path VARCHAR(255) NOT NULL,
        offset INTEGER NOT NULL
      )
    ''')
    c.execute('''\
      CREATE UNIQUE INDEX IF NOT EXISTS event_logs_index
      ON event_logs (run_id, path)
    ''')
