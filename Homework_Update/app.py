# -*- coding:utf-8 -*-

import os
from flask import Flask, request, jsonify

from lib.model import model_predict, model_load

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
  result = {}
  prefix = request.json['prefix']
  data_dir = request.json['data_dir']
  country = request.json.get('country','all')
  year = request.json['year']
  month = request.json['month']
  day = request.json['day']
  all_data, all_models = model_load(prefix=prefix,data_dir=data_dir)
  _result = model_predict(
    country,
    year,
    month,
    day,
    all_models=all_models,
    all_data=all_data,
    test=False
  )
  result['y_pred'] = _result['y_pred'].tolist() if _result['y_pred'] is not None else None
  result['y_proba'] = _result['y_proba'].tolist() if _result['y_proba'] is not None else None
  return jsonify(result)


if __name__ == '__main__':
  app.run(host=os.environ.get('HOST','0.0.0.0'),port=int(os.environ.get('PORT',5000)))
