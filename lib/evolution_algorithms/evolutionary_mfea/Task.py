import numpy as np
from lib.evolution_algorithms.evolutionary_mfea.Flatten import flatten, unflatten, shape_to_dims

class Task:
    """
    Task in MFEA that is performed by individuals
    """
    def __init__(self, name=None, item=None, model_shape=None, fnc=None, Lb=None, Ub=None):
        self.name = name
        self.item = item
        self.model_shape = model_shape
        self.fnc = fnc
        self.dims = shape_to_dims(self.model_shape)
        
        if Lb is None:
            self.Lb = -3 * np.ones(self.dims)
        else: 
            self.Lb = Lb
            
        if Ub is None:
            self.Ub = 3 * np.ones(self.dims)
        else:
            self.Ub = Ub

    # def fit_with_lstm(self, item=None, weights=None, cloud_metrics=None,  fitness_type=None):
    def fitness_evaluate(self, weights):
        cloud_metrics = {
            'train_data_type': self.name,
            'predict_data': self.name
        }
        weights_reshaped = unflatten(weights, self.model_shape)
        fitness_value, predictor = self.fnc(item=self.item, weights=weights_reshaped, cloud_metrics=cloud_metrics, fitness_type='bayesian_autoscaling')
        return fitness_value
        
    def __repr__(self):
        return "{}(Dims={}, Shape={}, Function={}, Lowerbound={}, Upperbound={})".format(self.name, self.dims, self.model_shape, self.fnc.__doc__, self.Lb, self.Ub)
    
    def __str__(self):
        return "{}(Dims={}, Shape={}, Function={}, Lowerbound={}, Upperbound={})".format(self.name, self.dims, self.model_shape, self.fnc.__doc__, self.Lb, self.Ub)