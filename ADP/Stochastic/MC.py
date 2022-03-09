# 循环中的场景生成
# 1）Guass 分布
def Guass_Generation (original, mean, variance):
    import numpy as np
    import math
    from random import gauss

    random_numbers = []
    for i in range(96):
        random_numbers.append(gauss(mean, math.sqrt(variance)))
    random_numbers = np.array(random_numbers)
    random_scences = np.sum([original, random_numbers], axis=0)

    return random_scences

# 2）Weibull 分布
def Weibull_Generation (original, shape, scale):
    import numpy as np

    random_numbers = scale * np.random.weibull(shape,96)-0.3
    random_scences = np.sum([original, random_numbers], axis=0)

    return random_scences
