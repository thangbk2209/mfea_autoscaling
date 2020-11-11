import numpy as np
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from config import *
from lib.preprocess.read_data import DataReader
from lib.scaler.preprocessing_data.data_normalizer import DataNormalizer


class DataPreprocessor:
    def __init__(self):
        self.train_size = Config.TRAIN_SIZE
        self.valid_size = Config.VALID_SIZE
        self.google_trace_config = Config.GOOGLE_TRACE_DATA_CONFIG
        self.read_data()

    def read_data(self):
        self.data = None
        data_reader = DataReader()
        official_data = data_reader.read()
        self.x_data, self.y_data = self.create_x_y_data(official_data)

    def create_x_y_data(self, official_data):

        if Config.DATA_EXPERIMENT == 'google_trace':
            # DEFINE X DATA
            if self.google_trace_config['train_data_type'] == 'cpu_mem':
                x_data = [official_data['cpu'], official_data['mem']]
            elif self.google_trace_config['train_data_type'] == 'cpu':
                x_data = [official_data['cpu']]
            elif self.google_trace_config['train_data_type'] == 'mem':
                x_data = [official_data['mem']]

            # DEFINE Y DATA
            if self.google_trace_config['predict_data'] == 'cpu':
                y_data = official_data['cpu']
            elif self.google_trace_config['train_data_type'] == 'mem':
                y_data = official_data['mem']

        else:
            print('|-> ERROR: Not support these data')

        return x_data, y_data

    def create_timeseries(self, X):
        if len(X) > 1:
            data = np.concatenate((X[0], X[1]), axis=1)
            if(len(X) > 2):
                for i in range(2, len(X), 1):
                    data = np.column_stack((data, X[i]))
        else:
            data = []
            for i in range(len(X[0])):
                data.append(X[0][i])
            data = np.array(data)
        return data

    def create_x(self, timeseries, sliding):
        dataX = []
        for i in range(len(timeseries) - sliding):
            datai = []
            for j in range(sliding):
                datai.append(timeseries[i + j])
            dataX.append(datai)
        return dataX

    def init_data_lstm(self, sliding, scaler_method):
        #print('>>> start init data for training LSTM model <<<')
        data_normalizer = DataNormalizer(scaler_method)
        x_timeseries, y_time_series, self.y_scaler = data_normalizer.normalize(
            self.x_data, self.y_data)

        num_points = x_timeseries.shape[0]
        train_point = int(self.train_size * num_points)

        x_sample = self.create_x(x_timeseries, sliding)

        x_train = x_sample[0:train_point - sliding]
        x_train = np.array(x_train)

        x_test = x_sample[train_point - sliding:]
        x_test = np.array(x_test)

        y_train = y_time_series[sliding: train_point]
        y_train = np.array(y_train)

        y_test = self.y_data[train_point:]
        y_test = np.array(y_test)

        # print(x_train.shape, x_test.shape)
        # print(y_train.shape, y_test.shape)
        # print('>>> Init data for training model complete <<<')

        return x_train, y_train, x_test, y_test, data_normalizer

    def init_data_ann(self, sliding, scaler_method):
        print('>>> start init data for training ANN model <<<')

        data_normalizer = DataNormalizer(scaler_method)
        x_timeseries, y_time_series, self.y_scaler = data_normalizer.normalize(
            self.x_data, self.y_data)

        num_points = x_timeseries.shape[0]
        train_point = int(self.train_size * num_points)

        x_sample = self.create_x(x_timeseries, sliding)

        x_train = x_sample[0:train_point - sliding]
        x_train = np.array(x_train)

        x_train = np.reshape(
            x_train, (x_train.shape[0], sliding * int(x_train.shape[2])))

        x_test = x_sample[train_point - sliding:]
        x_test = np.array(x_test)
        x_test = np.reshape(
            x_test, (x_test.shape[0], sliding * int(x_test.shape[2])))

        y_train = y_time_series[sliding: train_point]
        y_train = np.array(y_train)

        y_test = self.y_data[train_point:]
        y_test = np.array(y_test)

        return x_train, y_train, x_test, y_test, data_normalizer
