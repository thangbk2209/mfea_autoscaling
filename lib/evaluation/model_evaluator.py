import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.scaler.model_training import ModelTrainer
from config import *


class ModelEvaluator:
    def __init__(self):
        self.model_trainer = ModelTrainer()

    def evaluate_ann(self, iteration):
        pass

    def evaluate(self, x_train, y_train, x_test, y_test, data_normalizer, predictor, cloud_metrics):
        validation_point = int(Config.VALID_SIZE * x_train.shape[0])
        x_valid = x_train[validation_point:]
        y_valid = y_train[validation_point:]
        fitness, validation_error = self.model_trainer.fitness_manager.evaluate(predictor, data_normalizer, x_valid, y_valid)

        # predict_metric = cloud_metrics['predict_data']
        # best_folder_path = f'{Config.RESULTS_SAVE_PATH}/{predict_metric}/best_models'

        # model_name = predictor.model_path.split('/')[-1]
        # best_model_path = f'{best_folder_path}/{model_name}'
        # gen_folder_in_path(best_model_path)
        # predictor.save_model(best_model_path)
        
        # best_result_path = f'{best_folder_path}/results'
        # gen_folder_in_path(best_result_path)

        y_predict = predictor.predict(x_test)
        y_predict = data_normalizer.invert_tranform(y_predict)
        scaling_gap = np.full(y_predict.shape, validation_error)
        upper_y_predict = np.add(y_predict, scaling_gap)

        real_predict = np.concatenate((y_predict, y_test), axis=1)
        real_predict = np.concatenate((real_predict, upper_y_predict), axis=1)
        print('=== start plot ===')
        plt.plot(upper_y_predict)
        plt.plot(y_test)
        plt.show()

        prediction_df = pd.DataFrame(real_predict)
        # prediction_df.to_csv(f'{best_result_path}/prediction.csv', index=False, header=None)

        error = predictor.evaluate(x_test, y_test, data_normalizer)  

        errors = np.array([error['mae'], error['rmse'], error['mse'], error['mape'], error['smape'], fitness, validation_error])
        print(errors)
        # errors_df = pd.DataFrame(errors)
        # errors_df.to_csv(f'{best_result_path}/errors.csv',index=False, header=None)

    def evaluate_lstm(self, iteration):
        mem_cloud_metrics = {
            'train_data_type': 'mem',
            'predict_data': 'mem'
        }
        item_mem = {
            'scaler': 1,
            'sliding': 4,
            'batch_size': 64,
            'network_size': 2,
            'layer_size': 4,
            'activation': 1,
            'optimizer': 1,
            'dropout': 0.1,
            'learning_rate': 3e-4
        }
        lstm_predictor_mem, x_train_mem, y_train_mem, x_test_mem, y_test_mem, mem_data_normalizer = \
            self.model_trainer.build_lstm(item_mem, cloud_metrics=mem_cloud_metrics)

        cpu_cloud_metrics = {
            'train_data_type': 'cpu',
            'predict_data': 'cpu'
        }
        item_cpu = {
            'scaler': 1,
            'sliding': 4,
            'batch_size': 64,
            'network_size': 2,
            'layer_size': 8,
            'activation': 1,
            'optimizer': 1,
            'dropout': 0.1,
            'learning_rate': 3e-4
        }
        lstm_predictor_cpu, x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu, cpu_data_normalizer = \
            self.model_trainer.build_lstm(item_cpu, cloud_metrics=cpu_cloud_metrics)
        if Config.RUN_OPTION == 1:
            # mem model - ga
            # load weight
            weight_path = '/Users/thangnguyen/working/hust/bkc/research/data_science/mfea_lstm/data/mfea_result/mem/gen_2/mem'
            with open(weight_path, 'rb') as fp:
                weights_mem = pickle.load(fp)
            print(weights_mem)

            lstm_predictor_mem.set_weights(weights_mem)
            self.evaluate(x_train_mem, y_train_mem, x_test_mem, y_test_mem, mem_data_normalizer, lstm_predictor_mem, mem_cloud_metrics)

        elif Config.RUN_OPTION == 2:
            # cpu model - ga
            weight_path = '/Users/thangnguyen/working/hust/bkc/research/data_science/mfea_lstm/data/mfea_result/cpu/gen_199/cpu'
            with open(weight_path, 'rb') as fp:
                weights_cpu = pickle.load(fp)
            lstm_predictor_cpu.set_weights(weights_cpu)
            self.evaluate(x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu, cpu_data_normalizer, lstm_predictor_cpu, cpu_cloud_metrics)
        elif Config.RUN_OPTION == 12:
            # mem-cpu model - mfea
            mem_weight_path = '/Users/thangnguyen/working/hust/bkc/research/data_science/mfea_lstm/data/mfea_result/mem_cpu/gen_199/mem'
            with open(mem_weight_path, 'rb') as fp:
                weights_mem = pickle.load(fp)
            lstm_predictor_mem.set_weights(weights_mem)
            self.evaluate(x_train_mem, y_train_mem, x_test_mem, y_test_mem, mem_data_normalizer, lstm_predictor_mem, mem_cloud_metrics)
            
            cpu_weight_path = '/Users/thangnguyen/working/hust/bkc/research/data_science/mfea_lstm/data/mfea_result/mem_cpu/gen_199/cpu'
            with open(cpu_weight_path, 'rb') as fp:
                weights_cpu = pickle.load(fp)
            lstm_predictor_cpu.set_weights(weights_cpu)
            self.evaluate(x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu, cpu_data_normalizer, lstm_predictor_cpu, cpu_cloud_metrics)
        else:
            print('=== ERROR ===')
