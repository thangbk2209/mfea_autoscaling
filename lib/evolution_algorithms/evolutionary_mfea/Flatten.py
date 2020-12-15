import numpy as np
from functools import reduce
from itertools import chain

def _flatten(values):
    if isinstance(values, np.ndarray):
        yield values.flatten()
    else:
        for value in values:
            yield from _flatten(value)

def flatten(values):
    # flatten nested lists of np.ndarray to np.ndarray
    return np.concatenate(list(_flatten(values)))

def _unflatten(flat_values, prototype, offset):
    if isinstance(prototype, np.ndarray):
        shape = prototype.shape
        new_offset = offset + np.product(shape)
        value = flat_values[offset:new_offset].reshape(shape)
        return value, new_offset
    else:
        result = []
        for value in prototype:
            value, offset = _unflatten(flat_values, value, offset)
            result.append(value)
        return result, offset

def unflatten(flat_values, model_shape):
    # unflatten np.ndarray to nested lists with structure of prototype
    prototype = build_prototype(model_shape)
    result, offset = _unflatten(flat_values, prototype, 0)
    assert(offset == len(flat_values))
    return result

def build_prototype(model_shape):
    # if isinstance(model_shape, tuple):
    #     return np.ones(model_shape)
    if isinstance(model_shape, np.ndarray):
        return np.ones(tuple(model_shape))
    elif isinstance(model_shape, list):
        return [build_prototype(item) for item in model_shape]

def shape_to_dims(model_shape):
    if isinstance(model_shape, list):
        if len(model_shape) == 1:
            return shape_to_dims(model_shape[0])
        return reduce(lambda x, y: shape_to_dims(x) + shape_to_dims(y), model_shape)
    # if isinstance(model_shape, tuple):
    #     if len(model_shape) == 1:
    #         return model_shape[0]
    #     return reduce(lambda x, y: x * y, model_shape)
    if isinstance(model_shape, np.ndarray):
        if np.size(model_shape) == 1:
            return model_shape[0]
        # return reduce(lambda x, y: x * y, model_shape)
        return model_shape[0] * model_shape[1]
    return model_shape

# aa = [[np.array([[-0.50586534,  1.1574683 , -0.39299482,  0.852539  , -1.8036959 ,
#         -1.3033731 ,  2.8426437 ,  1.4165757 ,  0.33424678, -0.8826167 ,
#         -0.5805602 ,  0.18917157,  2.09449   ,  0.8436621 , -1.327958  ,
#          1.5975745 ]]), np.array([[-0.06867599,  0.25622413, -0.27193087, -0.08266257, -0.03697381,
#          0.0135735 ,  0.1054453 ,  0.05990096,  0.38690495, -0.1926048 ,
#         -0.48992762,  0.1760227 , -0.19524942, -0.491235  ,  0.15809707,
#          0.26525688],
#        [ 0.20207778,  0.01142468,  0.01228687,  0.07361928,  0.13213414,
#         -0.17975137,  0.3270569 ,  0.16481765,  0.28993034, -0.10422458,
#          0.00139803, -0.01792304, -0.52460825,  0.13003331, -0.44973996,
#         -0.42417932],
#        [ 0.06788754, -0.23150174, -0.24406624, -0.11044975, -0.29814622,
#         -0.17629397, -0.52853477, -0.3236758 ,  0.25702775,  0.18338546,
#          0.04253189, -0.12001606,  0.01442112, -0.23560649, -0.43896237,
#         -0.04039994],
#        [ 0.14149499,  0.5737325 ,  0.3831579 ,  0.0313249 , -0.21379   ,
#          0.01921816, -0.10292037,  0.04253399, -0.2952892 ,  0.2373085 ,
#         -0.310582  , -0.34758988, -0.09854607, -0.07778043, -0.24899998,
#          0.07738703]]), np.array([0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
#       )], [np.array([[-0.2563856 , -0.21327983, -0.03184087,  0.67222285, -1.0678334 ,
#          0.14620331, -0.76803184, -1.0599524 ],
#        [ 0.42002437,  0.07116603, -0.10035843,  0.5966408 , -0.41371173,
#          0.57214725, -0.0778802 ,  0.7637921 ],
#        [ 0.07816524, -0.7244629 ,  0.10575643,  0.35672343, -0.0480341 ,
#          0.90150917, -0.5925294 ,  0.16127452],
#        [ 0.76981646,  0.766314  , -0.73442805,  0.13466501, -0.48114577,
#          1.239446  ,  1.2871525 , -1.1577183 ]]), np.array([[-0.0776782 , -0.1116179 , -0.15893398,  0.11099251, -0.4400083 ,
#          0.7134914 , -0.38200524,  0.30874413],
#        [ 0.40913308, -0.3133576 ,  0.69168854,  0.11953397,  0.04315463,
#         -0.14902273, -0.2592978 ,  0.38780186]]), np.array([0., 0., 1., 1., 0., 0., 0., 0.])], [np.array([[-0.7680801],
#        [ 0.6286975]]), np.array([0.])]]

# model_shape = [[j.shape for j in i] for i in aa]
# print(type(aa[0][0].shape))
# bb = flatten(aa)
# print(bb)
# print(type(bb))
# cc = unflatten(bb, model_shape)
# print(aa)
# print(cc)
# print(model_shape)
# aa = [[np.array([ 1, 16]), np.array([ 4, 16]), np.array([16])], [np.array([4, 8]), np.array([2, 8]), np.array([8])], [np.array([2, 1]), np.array([1])]]
