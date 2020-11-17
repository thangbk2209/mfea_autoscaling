# example of bayesian optimization for a 1d function from scratch
import math
from math import pi
from operator import indexOf
import numpy as np
from numpy import arange
from numpy import vstack
from numpy import argmax,argmin
from mpl_toolkits.mplot3d import Axes3D
from lib.includes.utility import *
from config import *
from numpy import asarray
from numpy.core.fromnumeric import argmin
from numpy.random import normal
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
#k = GPflow.kernels.Matern32(1, variance=1, lengthscales=1.2)
# objective function
class GaussProcess:

    def __init__(self,objective):
        self.objective=objective
        self.X=[]
        self.y=[]
        self.name=[]
        self._parse_domain()

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
        Xsample=[]
        for index,value in enumerate(type_attr):
            
            if value == 'discrete':
                X_ = np.random.choice(range_val[index])
                Xsample.append(X_)
            if value == 'continuous':
                X_ = np.round(np.random.rand()*(max_val[index]-min_val[index])+min_val[index],5)
                Xsample.append(X_)

        self.name = names
        
        self.type_attr = type_attr
        self.max_val = np.array(max_val)
        self.min_val = np.array(min_val)
        self.range_val = range_val
        #print(Xsample)
        self.X.append(Xsample)
        #print(self.decode_position(Xsample))
        self.y.append(self.objective(self.decode_position(Xsample))[0])

    def decode_position(self, position):
        result = {}
        for i,name in enumerate(self.name):
            result[name] = position[i]
        return result
    # surrogate or approximation for the objective function
    def surrogate(self,model, X):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return model.predict(X, return_std=True)

    # probability of improvement acquisition function
    def acquisition(self, X, Xsamples, model):
        # calculate the best surrogate score found so far
        yhat, _ = self.surrogate(model, X)
        best = min(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(model, Xsamples)
        #print(mu)
        try:
            mu = mu[:, 0]
        except:
            mu=mu
        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std+1E-9))
        return probs

    def opt_acquisition(self, X, y, model):
        # random search, generate random samples

        Xsamples = []
        for j in range(100):
            Xsample = []
            for index,value in enumerate(self.type_attr):
                if value == 'discrete':
                    X_ = np.random.choice(self.range_val[index])
                    Xsample.append(X_)
                if value == 'continuous':
                    X_ = np.round(np.random.rand()*(self.max_val[index]-self.min_val[index])+self.min_val[index],5)
                    Xsample.append(X_)
            Xsamples.append(Xsample)
        #print(Xsamples)

        #Xsamples = Xsamples.reshape(len(Xsamples),)
        # calculate the acquisition function for each sample
        scores = self.acquisition(X, Xsamples, model)
        #print(scores.shape)
        # locate the index of the largest scores
        #print(scores)
        #print(Xsamples)
        ix = argmin(scores)
        print(ix)
        #print(Xsamples[ix])
        #print(Xsamples[ix])
        return Xsamples[ix]

    def fit(self):
        model = GaussianProcessRegressor()
        for i in range(10):
        # select the next point to sample
            x = self.opt_acquisition(self.X,self.y, model)
        # sample the point
            actual = self.objective(self.decode_position(x))[0]
        # summarize the finding
            est, _ = self.surrogate(model, [x])
            #print(self.X)
            print('>x1={},x2={}, f()={}, actual={}'.format(x[0],x[1], est, actual))
        # add the data to the dataset
            if not math.isnan(actual):
                self.X = vstack((self.X, [x]))
                self.y = vstack((self.y, [actual]))
            else:
                pass
        # update the model
            model.fit(self.X, self.y)
        ix = argmin(self.y)
        print('Best Result: x1=%.3f,x2=%3f, y=%.3f' % (self.X[ix][0],self.X[ix][1], self.y[ix]))
        return self.X[ix]