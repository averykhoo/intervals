import sys; sys.path[:0] = ['C:/Users/user/PycharmProjects/intervals/archive/v1']
import random, multi_interval as v1
random.seed(0)
print('calling random_multi_interval(0, 0.5, 5, decimals=1) ...', flush=True)
print(v1.random_multi_interval(0, 0.5, 5, 1))
