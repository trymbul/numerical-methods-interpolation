import numpy as np
def Runge_func(x):
    return 1 / (1 + x**2)

def cos_func(x):
    return np.cos(2 * np.pi * x)

def e_func(x):
    return np.e**(3*x) * np.sin(2*x)