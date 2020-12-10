import numpy as np
from functools import reduce
from itertools import chain
    
def flatten(arr):
    l1 = [len(s) for s in chain.from_iterable(arr)]
    l2 = [len(s) for s in arr]
    arr = np.array(list(chain.from_iterable(
        chain.from_iterable(arr))))
    return arr,l1,l2

def reconstruct(arr,l1,l2):
    arr_split = np.split(np.split(arr,np.cumsum(l1)),np.cumsum(l2))[:-1]
    result = np.array([reduce(lambda x, y: np.vstack((x, y)), item) for item in arr_split])
    return result

def check_equal(arr1, arr2):
    if 1: 
        pass
    return np.array([check_equal(arr1[i], arr2[i]) for i in range(len(arr1))]).all()