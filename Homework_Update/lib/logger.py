# -*- coding: utf-8 -*-

"""
module with functions to enable logging
"""

import time, os, csv, uuid
from datetime import date

LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, 'logs')

if not os.path.exists(LOGDIR):
  os.mkdir(LOGDIR)

def update_train_log(
  tag,
  data_shape,
  eval_test,
  runtime,
  model_version,
  model_version_note,
  test=False
):
  """
  update train log file
  """

  # name the logfile using something that cycles with date (day, month, year)
  today = date.today()
  if test:
    logfile = os.path.join(LOGDIR, "train-test.csv")
  else:
    logfile = os.path.join(
      LOGDIR, "train-{}-{}.csv".format(today.year, today.month)
    )

  # write the data to a csv file
  header = [
    'tag', 'unique_id', 'timestamp', 'x_shape', 'eval_test', 'model_version',
    'model_version_note', 'runtime'
  ]
  write_header = False
  if not os.path.exists(logfile):
    write_header = True
  with open(logfile, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    if write_header:
      writer.writerow(header)

    to_write = map(
      str, [
        tag,
        uuid.uuid4(),
        time.time(), data_shape, eval_test, model_version, model_version_note,
        runtime
      ]
    )
    writer.writerow(to_write)


def update_predict_log(
  y_pred, y_proba, query, runtime, model_version, test=False
):
  """
  update predict log file
  """

  # name the logfile using something that cycles with date (day, month, year)
  today = date.today()
  if test:
    logfile = os.path.join(LOGDIR, "predict-test.csv")
  else:
    logfile = os.path.join(
      LOGDIR, "predict-{}-{}.csv".format(today.year, today.month)
    )

  # write the data to a csv file
  header = [
    'unique_id', 'timestamp', 'y_pred', 'y_proba', 'query', 'model_version',
    'runtime'
  ]
  write_header = False
  if not os.path.exists(logfile):
    write_header = True
  with open(logfile, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    if write_header:
      writer.writerow(header)

    to_write = map(
      str, [
        uuid.uuid4(),
        time.time(), y_pred, y_proba, query, model_version, runtime
      ]
    )
    writer.writerow(to_write)


if __name__ == "__main__":
  """
  basic test procedure for logger.py
  """

  from lib.model import MODEL_VERSION, MODEL_VERSION_NOTE

  # train logger
  update_train_log(
    'all',
    str((100, 10)),
    "0.5",
    "00:00:01",
    MODEL_VERSION,
    MODEL_VERSION_NOTE,
    test=True
  )
  # predict logger
  update_predict_log(
    "[0]",
    "[0.6,0.4]",
    "['united_states', 24, 'aavail_basic', 8]",
    "00:00:01",
    MODEL_VERSION,
    test=True
  )
