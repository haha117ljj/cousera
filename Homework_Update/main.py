# -*- coding:utf-8 -*-

import argparse

from lib.model import model_predict, model_load  # model_train

def predict(args):
  all_data, all_models = model_load(prefix=args.prefix, data_dir=args.data_dir)
  result = model_predict(
    args.country,
    args.year,
    args.month,
    args.day,
    all_models=all_models,
    all_data=all_data,
    test=True
  )
  return result

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-p','--prefix',default='test')
  parser.add_argument('-d','--data_dir',default='test')
  parser.add_argument('-c','--country',default='united_kingdom')
  parser.add_argument('-Y','--year',default='2018')
  parser.add_argument('-M','--month',default='01')
  parser.add_argument('-D','--day',default='05')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  result = predict(args)
  print(result)
