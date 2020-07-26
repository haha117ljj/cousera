# -*- coding: utf-8 -*-

'''
Test module for API, Model, Logger
'''

import unittest

from lib.model import MODEL_VERSION, MODEL_VERSION_NOTE, model_train, model_load, model_predict
from lib.logger import update_predict_log, update_train_log
from lib.cslib import fetch_ts

class TestApi(unittest.TestCase):
  def test_load_data(self):
    datadir = 'test/original'
    country = {
      'all','united_kingdom','poland','france',
      'germany','portugal','australia','usa','eire','belgium','netherlands'
    }
    try:
      dfs = fetch_ts(datadir)
    except BaseException:
      self.fail('fetch_data execution failed.')
    else:
      self.assertEqual(set(dfs.keys()),country)

class TestModel(unittest.TestCase):
  def test_model_train(self):
    data_dir = './test/'
    try:
      model_train(data_dir, test=True)
    except BaseException:
      self.fail('model train failed.')

  def test_model_load(self):
    data_dir = './test/'
    try:
      _, _ = model_load(prefix='None',data_dir='')
    except BaseException as e:
      self.assertEqual('{}'.format(e),'Models with prefix \'None\' cannot be found did you train?')
    try:
      _, _ = model_load(prefix='test',data_dir=data_dir,training=False)
    except BaseException:
      self.fail('model load failed.')

  def test_model_predict(self):
    country = 'united_kingdom'
    year = '2018'
    month = '01'
    day = '05'
    data_dir = './test/'
    try:
      all_data, all_models = model_load(prefix='test',data_dir=data_dir,training=False)
      result = model_predict(country,year,month,day,all_models=all_models,all_data=all_data,test=True)
    except BaseException:
      self.fail('model predict failed.')
    else:
      if 'y_pred' not in result or 'y_proba' not in result:
        self.fail('model predicted result must include \'y_pred\' and \'y_proba\'.')

class TestLogger(unittest.TestCase):
  def test_update_predict_log(self):
    try:
      update_predict_log(
        "[0]",
        None,
        "2020-07-23",
        "000:00:01",
        MODEL_VERSION,
        test=True
      )
    except BaseException:
      self.fail('update_predict_log call failed.')

  def test_update_train_log(self):
    try:
      update_train_log(
        'all',
        str((100, 10)),
        "0.5",
        "000:00:01",
        MODEL_VERSION,
        MODEL_VERSION_NOTE,
        test=True
      )
    except BaseException:
      self.fail('update_train_log call failed.')
