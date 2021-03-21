from lib.evolution_algorithms.evolutionary_mfea.Task import Task

def decode(Task, rnvec):
    d = Task.dims
    nvars = rnvec[:d]
    minrange = Task.Lb[:d]
    maxrange = Task.Ub[:d]
    y = maxrange - minrange
    variables = y * (nvars + 1)/2 + minrange
    return variables