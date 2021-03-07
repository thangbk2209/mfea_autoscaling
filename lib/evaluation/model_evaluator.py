from lib.scaler.model_training import ModelTrainer
from config import *


class ModelEvaluator:
    def __init__(self):
        self.model_trainer = ModelTrainer()

    def evaluate_ann(self, iteration):
        pass

    def evaluate(self, x_train, y_train, predictor):
        validation_point = int(Config.VALID_SIZE * x_train.shape[0])
        x_valid = x_train[validation_point:]
        y_valid = y_train[validation_point:]
        fitness, validation_error = self.model_trainer.fitness_manager.evaluate(lstm_predictor, data_normalizer, x_valid, y_valid)

        predict_metric = cloud_metrics['predict_data']
        best_folder_path = f'{self.results_save_path}/{predict_metric}/best_models'

        model_name = lstm_predictor.model_path.split('/')[-1]
        best_model_path = f'{best_folder_path}/{model_name}'
        gen_folder_in_path(best_model_path)
        lstm_predictor.save_model(best_model_path)
        
        best_result_path = f'{best_folder_path}/results'
        gen_folder_in_path(best_result_path)

        y_predict = lstm_predictor.predict(x_test)
        y_predict = data_normalizer.invert_tranform(y_predict)
        scaling_gap = np.full(y_predict.shape, validation_error)
        upper_y_predict = np.add(y_predict, scaling_gap)

        real_predict = np.concatenate((y_predict, y_test), axis=1)
        real_predict = np.concatenate((real_predict, upper_y_predict), axis=1)
        prediction_df = pd.DataFrame(real_predict)
        prediction_df.to_csv(f'{best_result_path}/prediction.csv', index=False, header=None)

        error = lstm_predictor.evaluate(x_test, y_test, data_normalizer)  

        errors = np.array([error['mae'], error['rmse'], error['mse'], error['mape'], error['smape'], fitness, validation_error])
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(f'{best_result_path}/errors.csv',index=False, header=None)

    def evaluate_lstm(self, iteration):
        mem_cloud_metrics = {
            'train_data_type': 'mem',
            'predict_data': 'mem'
        }
        item_mem = {
            'scaler': 1,
            'sliding': 2,
            'batch_size': 64,
            'network_size': 2,
            'layer_size': 4,
            'activation': 0,
            'optimizer': 0,
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
            'network_size': 3,
            'layer_size': 8,
            'activation': 0,
            'optimizer': 0,
            'dropout': 0.1,
            'learning_rate': 3e-4
        }
        lstm_predictor_cpu, x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu, cpu_data_normalizer = \
            self.model_trainer.build_lstm(item_cpu, cloud_metrics=cpu_cloud_metrics)
        if Config.RUN_OPTION == 1:
            # mem model - ga
            # load weight
            weights_mem = None
            lstm_predictor_mem.set_weights(weights_mem)

            

        elif Config.RUN_OPTION == 2:
            # cpu model - ga
            weights_cpu = None
            lstm_predictor_cpu.set_weights(weights_cpu)
        elif Config.RUN_OPTION == 12:
            # mem-cpu model - mfea
            weights_mem = None
            lstm_predictor_mem.set_weights(weights_mem)
            
            weights_cpu = None
            lstm_predictor_cpu.set_weights(weights_cpu)
        else:
            print('=== ERROR ===')
