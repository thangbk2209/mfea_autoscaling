import os
import sys

from config import *
from lib.scaler.model_training import ModelTrainer
from lib.evaluation.model_evaluator import ModelEvaluator


def init_model():
    print('[1] >>> Start init model')
    print(f'=== model information: {Config.RESULTS_SAVE_PATH}')
    model_trainer = ModelTrainer()
    model_trainer.train()
    print('[1] >>> Init model complete')


def evaluate_model():

    try:
        iteration = sys.argv[2]
        value_optimize = sys.argv[3]
        model_name = sys.argv[4]
    except Exception as ex:
        value_optimize = ''
        print('[ERROR] Can not define your iteration')

    model_evaluator = ModelEvaluator()

    if model_name == 'ann':
        model_evaluator.evaluate_ann(iteration)
    elif model_name == 'lstm':
        model_evaluator.evaluate_lstm(iteration)
    else:
        print(f'[ERROR] model not supported: {model_name}')


if __name__ == "__main__":
    if sys.argv[1] == 'training':
        init_model()
    elif sys.argv[1] == 'evaluate':
        evaluate_model()
    else:
        print(f'[ERROR] Not support: {sys.argv[1]}')
