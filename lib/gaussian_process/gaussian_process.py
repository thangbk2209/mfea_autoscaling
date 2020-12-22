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
        self.estimate=[0]
        self.cloud_metrics = {
                'train_data_type': 'cpu',
                'predict_data': 'cpu'
            }
        self.alpha = Config.ALPHA
        self._parse_domain()
        self.population_size = Config.POPULATION_SIZE
        self.max_iteration = Config.MAX_ITER

    def gen_sample(self):
        x_sample = []
        for index, value in enumerate(self.type_attr):
            if value == 'discrete':
                _x = (np.random.choice(self.range_val[index])-self.min_val[index])/(self.max_val[index]-self.min_val[index])
                #print(_x)
                x_sample.append(_x)
                #x_sample_memory.append(_x)

            if value == 'continuous':
                # _old_x = self.min_val + (self.max_val - self.min_val) * np.random.rand(len(self.type_attr))
                # _x = np.round(np.random.rand() * (self.max_val[index] - self.min_val[index]) + self.min_val[index], 5)
                _x = (np.random.rand() * (self.max_val[index] - self.min_val[index]))/(self.max_val[index]-self.min_val[index])
                x_sample.append(_x)
                #x_sample_memory.append(_x)

            if self.name[index] in ["sliding","network_size","layer_size"]:
                if value == 'discrete':
                    _x = (np.random.choice(self.range_val[index])-self.min_val[index])/(self.max_val[index]-self.min_val[index])
                    x_sample.append(_x)
                if value == 'continuous':
                # _old_x = self.min_val + (self.max_val - self.min_val) * np.random.rand(len(self.type_attr))
                # _x = np.round(np.random.rand() * (self.max_val[index] - self.min_val[index]) + self.min_val[index], 5)
                    _x = (np.random.rand() * (self.max_val[index] - self.min_val[index]))/(self.max_val[index]-self.min_val[index])
                    x_sample.append(_x)
                #print(x_sample)
        return x_sample

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
                min_val.append(attr['domain'][0])
                max_val.append(attr['domain'][len(attr['domain']) - 1])
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
        self.x.append(x_sample)
        
        x_cpu,x_mem=self.split_sample(x_sample)
        # @TODO thangbk2209 need to add fitness_type and cloud_metrics into objective_function
        self.y.append(self.alpha*self.objective_function(self.decode_sample(x_cpu),cloud_metrics=self.cloud_metrics)[0]+\
            (1-self.alpha)*self.objective_function(self.decode_sample(x_mem))[0])

    def split_sample(self,sample):
        x_cpu = []
        x_mem = []
        #print(sample)
        for i in range(len(sample)):
            if i in [0,1]:
                x_cpu.append(int(sample[i]*(self.max_val[i]-self.min_val[i]))+self.min_val[i])
                x_mem.append(int(sample[i]*(self.max_val[i]-self.min_val[i]))+self.min_val[i])
            elif i in [2,4,6]:
                x_cpu.append(int(sample[int(i-(i-2)/2)]*(self.max_val[int(i-(i-2)/2)]-self.min_val[int(i-(i-2)/2)]))+self.min_val[int(i-(i-2)/2)])
            elif i in [3,5,7]:
                x_mem.append(int(sample[int(i-1-(i-3)/2)]*(self.max_val[int(i-1-(i-3)/2)]-self.min_val[int(i-1-(i-3)/2)]))+self.min_val[int(i-1-(i-3)/2)])
            elif i in [8,9]:
                x_cpu.append(sample[i-3]*(self.max_val[i-3]-self.min_val[i-3])+self.min_val[i-3])
                x_mem.append(sample[i-3]*(self.max_val[i-3]-self.min_val[i-3])+self.min_val[i-3])
            else:
                x_cpu.append(int(sample[i-3]*(self.max_val[i-3]-self.min_val[i-3])))
                x_mem.append(int(sample[i-3]*(self.max_val[i-3]-self.min_val[i-3])))
        #print(x_cpu,x_mem)
        return x_cpu, x_mem

    def decode_sample(self, sample):
        result = {}
        for i, name in enumerate(self.name):
            if name in ["learning_rate","dropout"]:
                result[name] = sample[i]
            else:
                result[name]=int(sample[i])
        return result

    # surrogate or approximation for the objective function
    def surrogate(self, x):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter('ignore')
            return self.gaussian_process_model.predict(x, return_std=True)

    # probability of improvement acquisition function
    def acquisition(self, x, x_samples):
        # calculate the best surrogate score found so far
        yhat, _ = self.surrogate(x)
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

    def opt_acquisition(self, x):
        # random search, generate random samples

        x_samples = []
        for j in range(self.population_size):
            x_sample = self.gen_sample()
            x_samples.append(x_sample)
        #print(x[:,0])
        #print("_____________________________")
        #print(x_samples[:,0])
        # calculate the acquisition function for each sample
        scores = self.acquisition(x, x_samples)
        min_sample_idx = argmin(scores)
        #min_sample_idx2 = argmin(scores)

        return x_samples[min_sample_idx]

    def optimize(self):
        self.gaussian_process_model = GaussianProcessRegressor()
        #self.gaussian_process_model_mem = GaussianProcessRegressor()


        for i in range(self.max_iteration):

            # select the next point to sample
            x = self.opt_acquisition(self.x)

            # sample the point
            x_cpu, x_mem = self.split_sample(x)
            actual = self.alpha*self.objective_function(item=self.decode_sample(x_cpu),cloud_metrics=self.cloud_metrics)[0]\
                + (1-self.alpha)*self.objective_function(item=self.decode_sample(x_mem))[0]

            # summarize the finding
            est, _ = self.surrogate([x])
            #est1, _1 = self.surrogate([x[0]],type="cpu")

            print(est)
            print('>x={}, f()={}, actual={}'.format(x, est, actual))
            #print('>x1={},c f()={}, actual={}'.format(x[1], est1, actual))

            # add the data to the dataset
            if not math.isnan(actual):
                self.x = vstack((self.x, [x]))
                self.y = vstack((self.y, [actual]))
                self.estimate.append(est)
            # update the gausian model
            self.gaussian_process_model.fit(self.x, self.y)
            #self.gaussian_process_model_mem.fit(self.x[:,0], self.y[:,0])


        optimal_sample_idx = argmin(self.y)
        print(f'Best Result: x1={self.x[optimal_sample_idx][0]},x2={self.x[optimal_sample_idx][1]}, y={self.y[optimal_sample_idx]}')
        
        files = open("optimization_result.csv","w")
        files.write("x,y,actual\n")
        print(len(self.x))
        print(len(self.y))
        print(len(self.estimate))


        for i in range(len(self.y)):
            print(i)
            files.write("{},{},{}\n".format(self.x[i], self.y[i], self.estimate[i]))
        return self.x[optimal_sample_idx]
