from math import factorial
import numpy as np

def max_norm_error(f, p, x):
    #Finding the error in every point
    error_array = np.abs(f(x)-p)
    #Returning the biggest error
    return np.max(error_array)

def two_norm_error(f, p, x, start, stop):
    #Approximating 2-norm error using the formula given in the task
    N = len(x)
    coeff = np.sqrt(stop - start) / np.sqrt(N)
    summand = (f(x) - p)**2
    return coeff * np.sqrt(np.sum(summand))

def error_bound_cos2pi(degree, nodes, x):
    n = degree + 1  #Finding the number of nodes
    M = (2 * np.pi)**n  #Computing the maximum of the derivative
    pi = np.ones_like(x)
    for node in nodes:
        pi *= np.abs(x - node)  #Constructing the nodal polynomial
    return M / factorial(n) * np.max(pi)

def piecewice_max_norm_error(f, lagranges):
    max_error = 0   #Initiating the variable
    for i in range(len(lagranges)):
        (p, x) = lagranges[i]   #Computing the approximation and domain for each sub-interval
        interval_error = max_norm_error(f, p, x)    #Finding the max-norm error

        #Finding the biggest of the max-norm errors i.e. the max-norm error of the whole domain
        if interval_error > max_error: 
            max_error = interval_error
    return max_error