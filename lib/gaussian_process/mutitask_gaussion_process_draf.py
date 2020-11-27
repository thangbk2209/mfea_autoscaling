# example of bayesian optimization for a 1d function from scratch
import math
from math import pi
from operator import indexOf

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import arange
from numpy import vstack
from numpy import argmax, argmin
from numpy import asarray
from numpy.core.fromnumeric import argmin
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, ConstantKernel
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot 
from skopt import gp_minimize

from lib.includes.utility import *
from config import *

class GaussProcess:

    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.x = []  # Chromosome that has fitness value
        self.y = []  # Fit value of chromosome in X
        self.name = []
        self.cloud_metrics = {
                'train_data_type': 'cpu',
                'predict_data': 'cpu'
            }
        self._parse_domain()
        self.population_size = Config.POPULATION_SIZE
        self.max_iteration = Config.MAX_ITER

    def gen_sample(self):
        x_sample_cpu = []
        x_sample_memory = []
        for index, value in enumerate(self.type_attr):
            if value == 'discrete':
                _x = np.random.choice(self.range_val[index])
                x_sample_cpu.append(_x)
                x_sample_memory.append(_x)

            if value == 'continuous':
                # _old_x = self.min_val + (self.max_val - self.min_val) * np.random.rand(len(self.type_attr))
                # _x = np.round(np.random.rand() * (self.max_val[index] - self.min_val[index]) + self.min_val[index], 5)
                _x = np.random.rand() * (self.max_val[index] - self.min_val[index]) + self.min_val[index]
                x_sample_cpu.append(_x)
                x_sample_memory.append(_x)

            if self.name[index] in ["sliding","network_size","layer_size"]:
                if value == 'discrete':
                    _x = np.random.choice(self.range_val[index])
                    x_sample_memory[-1]=_x
                if value == 'continuous':
                # _old_x = self.min_val + (self.max_val - self.min_val) * np.random.rand(len(self.type_attr))
                # _x = np.round(np.random.rand() * (self.max_val[index] - self.min_val[index]) + self.min_val[index], 5)
                    _x = np.random.rand() * (self.max_val[index] - self.min_val[index]) + self.min_val[index]
                    x_sample_memory[-1]=_x
        return x_sample_cpu, x_sample_memory

    def _parse_domain(self):
        domain = Config.LSTM_CONFIG['domain']
        names = []
        type_attr = []
        max_val = []
        min_val = []
        range_val = []
        for attr in domain:
            names.append(attr['name'])
            type_attr.append(attr['type'])
            if attr['type'] == 'discrete':
                min_val.append(0)
                max_val.append(len(attr['domain']) - 1)
            elif attr['type'] == 'continuous':
                min_val.append(attr['domain'][0])
                max_val.append(attr['domain'][1])
            range_val.append(attr['domain'])

        self.name = names
        
        self.type_attr = type_attr
        self.max_val = np.array(max_val)
        self.min_val = np.array(min_val)
        self.range_val = range_val

        x_sample = self.gen_sample()
        print(x_sample)
        self.x.append([x_sample[0],x_sample[1]])
        # @TODO thangbk2209 need to add fitness_type and cloud_metrics into objective_function
        self.y.append([self.objective_function(self.decode_sample(x_sample[0]),cloud_metrics=self.cloud_metrics)[0],\
            self.objective_function(self.decode_sample(x_sample[1]))[0]])

    def decode_sample(self, sample):
        result = {}
        for i, name in enumerate(self.name):
            if name in ["learning_rate","dropout"]:
                result[name] = sample[i]
            else:
                result[name]=int(sample[i])
        return result

    # surrogate or approximation for the objective function
    def surrogate(self, x, type=" "):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter('ignore')

            if type == "mem":
                return self.gaussian_process_model_mem.predict(x, return_std=True)
            else:
                return self.gaussian_process_model_cpu.predict(x, return_std=True)

    # probability of improvement acquisition function
    def acquisition(self, x, x_samples,type):
        # calculate the best surrogate score found so far
        yhat, _ = self.surrogate(x,type)
        best = min(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(x_samples)

        try:
            mu = mu[:, 0]
        except:
            mu = mu

        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std + 1E-9))
        return probs

    def opt_acquisition(self, x, y):
        # random search, generate random samples

        x_samples = []
        for j in range(100):
            x_sample = self.gen_sample()
            x_samples.append([x_sample[0],x_sample[1]])
        x_samples = np.array(x_samples)
        x = np.array(x)
        #print(x[:,0])
        #print("_____________________________")
        #print(x_samples[:,0])
        # calculate the acquisition function for each sample
        scores1 = self.acquisition(x[:,0], x_samples[:,0],type="mem")
        scores2 = self.acquisition(x[:,1], x_samples[:,1],type="cpu")
        min_sample_idx1 = argmin(scores1)
        min_sample_idx2 = argmin(scores2)

        return [x_samples[min_sample_idx1,0] , x_samples[min_sample_idx2,1]]

    def optimize(self):
        self.gaussian_process_model_cpu = GaussianProcessRegressor()
        self.gaussian_process_model_mem = GaussianProcessRegressor()


        for i in range(self.max_iteration):

            # select the next point to sample
            x = self.opt_acquisition(self.x, self.y)

            # sample the point
            actual = [self.objective_function(self.decode_sample(x[0]),cloud_metrics=self.cloud_metrics)[0],\
                self.objective_function(self.decode_sample(x[1]))[0]]

            # summarize the finding
            est, _ = self.surrogate([x[0]],type="mem")
            est1, _1 = self.surrogate([x[0]],type="cpu")


            print('>x1={}, f()={}, actual={}'.format(x[0], est, actual))
            print('>x1={}, f()={}, actual={}'.format(x[1], est1, actual))

            # add the data to the dataset
            self.x = vstack((self.x, [x]))
            self.y = vstack((self.y, [actual]))
            # update the gausian model
            self.gaussian_process_model_cpu.fit(self.x[:,1], self.y[:,1])
            self.gaussian_process_model_mem.fit(self.x[:,0], self.y[:,0])


        optimal_sample_idx = argmin(self.y)
        print(f'Best Result: x1={self.x[optimal_sample_idx][0]},x2={self.x[optimal_sample_idx][1]}, y={self.y[optimal_sample_idx]}')
        return self.x[optimal_sample_idx]
