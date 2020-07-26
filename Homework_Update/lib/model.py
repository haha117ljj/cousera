# -*- coding: utf-8 -*-

"""
Created on Wed Jul 15 22:50:36 2020

@author: 209949672
"""

import time, os, re, joblib

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor  # AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lib.logger import update_predict_log, update_train_log
from lib.cslib import fetch_ts, engineer_features

# model specific variables (iterate the version and note with each change)
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"


def _model_train(df, tag, test=False):
  """
  example funtion to train model

  The 'test' flag when set to 'True':
      (1) subsets the data and serializes a test version
      (2) specifies that the use of the 'test' log file
  """

  # start timer for runtime
  time_start = time.time()

  X, y, dates = engineer_features(df)

  if test:
    n_samples = int(np.round(0.3*X.shape[0]))
    subset_indices = np.random.choice(
      np.arange(X.shape[0]), n_samples, replace=False
    ).astype(int)
    mask = np.in1d(np.arange(y.size), subset_indices)
    y = y[mask]
    X = X[mask]
    dates = dates[mask]

  # Perform a train-test split
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42
  )
  # train a random forest model
  param_grid_rf = {
    'rf__criterion': ['mse', 'mae'],
    'rf__n_estimators': [10, 20, 50]
  }
  # param_grid_adb = {
  #   'adb__loss': ['linear', 'square', 'exponential'],
  #   'adb__n_estimators': [10, 15, 20, 25]
  # }
  # param_grid_svr = {
  #   'svr__kernel': ['linear', 'poly', 'rbf'],
  #   'svr__C': [0.01, 0.1, 1, 2, 19]
  # }

  pipe_rf = Pipeline(
    steps=[('scaler', StandardScaler()), ('rf', RandomForestRegressor())]
  )
  # pipe_adb = Pipeline(
  #   steps=[('scaler', StandardScaler()), ('adb', AdaBoostRegressor())]
  # )
  # pipe_svr = Pipeline(steps=[('scaler', StandardScaler()), ('svr', SVR())])

  grid = GridSearchCV(
    pipe_rf, param_grid=param_grid_rf, cv=5, iid=False, n_jobs=-1
  )
  grid.fit(X_train, y_train)
  y_pred = grid.predict(X_test)
  eval_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)))

  # retrain using all data
  grid.fit(X, y)
  model_name = re.sub(r"\.", "_", str(MODEL_VERSION))
  if test:
    saved_model = os.path.join(
      MODEL_DIR, "test-{}-{}.joblib".format(tag, model_name)
    )
    print("... saving test version of model: {}".format(saved_model))
  else:
    saved_model = os.path.join(
      MODEL_DIR, "sl-{}-{}.joblib".format(tag, model_name)
    )
    print("... saving model: {}".format(saved_model))

  joblib.dump(grid, saved_model)

  m, s = divmod(time.time() - time_start, 60)
  h, m = divmod(m, 60)
  runtime = "%03d:%02d:%02d" % (h, m, s)

  # update log
  update_train_log(
    tag,
    X.shape,
    eval_rmse,
    runtime,
    MODEL_VERSION,
    MODEL_VERSION_NOTE,
    test=test
  )


def model_train(data_dir, test=False):
  """
  funtion to train model given a df

  'mode' -  can be used to subset data essentially simulating a train
  """

  if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

  if test:
    print("... test flag on")
    print("...... subseting data")
    print("...... subseting countries")

  # fetch time-series formatted data
  ts_data = fetch_ts(data_dir)

  # train a different model for each data sets
  for country, df in ts_data.items():

    if test and country not in ['all', 'united_kingdom']:
      continue

    _model_train(df, country, test=test)


def model_load(prefix='sl', data_dir=None, training=True):
  """
  example funtion to load model

  The prefix allows the loading of different models
  """

  if not data_dir:
    data_dir = os.path.join(ROOT_DIR, "train_data")

  models = [
    f for f in os.listdir(MODEL_DIR)
    if re.match(r'^{}'.format(prefix), f)
  ]

  if len(models) == 0:
    raise Exception(
      "Models with prefix '{}' cannot be found did you train?".format(prefix)
    )

  all_models = {}
  for model in models:
    all_models[re.split("-", model)[1]
               ] = joblib.load(os.path.join(MODEL_DIR, model))

  # load data
  ts_data = fetch_ts(data_dir)
  all_data = {}
  for country, df in ts_data.items():
    X, y, dates = engineer_features(df, training=training)
    dates = np.array([str(d) for d in dates])
    all_data[country] = {"X": X, "y": y, "dates": dates}

  return (all_data, all_models)


def model_predict(
  country, year, month, day, all_models=None, all_data=None, test=False
):
  """
  example funtion to predict from model
  """
  # start timer for runtime
  time_start = time.time()

  # load model if needed
  if all_models is None or all_data is None:
    all_data, all_models = model_load(training=False)

  # input checks
  if country not in all_models.keys():
    raise Exception(
      "ERROR (model_predict) - model for country '{}' could not be found".
      format(country)
    )

  for d in [year, month, day]:
    if re.search(r"\D", d):
      raise Exception("ERROR (model_predict) - invalid year, month or day")

  # load data
  model = all_models[country]
  data = all_data[country]

  # check date
  target_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))

  if target_date not in data['dates']:
    raise Exception(
      "ERROR (model_predict) - date {} not in range {}-{}".format(
        target_date, data['dates'][0], data['dates'][-1]
      )
    )
  date_indx = np.where(data['dates'] == target_date)[0][0]
  query = data['X'].iloc[[date_indx]]

  # sainty check
  if data['dates'].shape[0] != data['X'].shape[0]:
    raise Exception("ERROR (model_predict) - dimensions mismatch")

  # make prediction and gather data for log entry
  y_pred = model.predict(query)
  y_proba = None
  if 'predict_proba' in dir(model) and 'probability' in dir(model):
    if model.probability:
      y_proba = model.predict_proba(query)

  m, s = divmod(time.time() - time_start, 60)
  h, m = divmod(m, 60)
  runtime = "%03d:%02d:%02d" % (h, m, s)

  # update predict log
  update_predict_log(
    y_pred, y_proba, target_date, runtime, MODEL_VERSION, test=test
  )

  return ({'y_pred': y_pred, 'y_proba': y_proba})


if __name__ == "__main__":
  """
  basic test procedure for model.py
  """

  # train the model
  print("TRAINING MODELS")
  data_dir = os.path.join('./train_data/')
  model_train(data_dir, test=False)
  '''
  # load the model
  print("LOADING MODELS")
  all_data, all_models = model_load(prefix='test',data_dir=os.path.join('.','train_data'))
  print("... models loaded: ",",".join(all_models.keys()))

  # test predict
  country = 'united_kingdom'
  year = '2018'
  month = '01'
  day = '05'
  result = model_predict(country,year,month,day,all_models=all_models,all_data=all_data,test=True)
  print(result)
  '''
