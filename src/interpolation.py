import numpy as np

def lagrange_basis_func(x, nodes, n, index):
    l_i = 1
    #constructing the lagrange basis for a node
    for j in range(n):
        if j == index: continue #skipping division by zero
        l_i *= (x - nodes[j])/(nodes[index] - nodes[j])
    return l_i

def lagrange_interpolation_polynomial(x, nodes, vals):
    num_nodes = len(nodes)
    if num_nodes != len(vals):
        raise ValueError("x and y must have the same length")

    p = 0   #the polynomial

    #summing the lagrange basis functions for all nodes
    for i in range(num_nodes):
        p += vals[i] * lagrange_basis_func(x, nodes, num_nodes, i)
    return p

def chebishev_nodes(n, start, stop):
    indices = np.arange(n)
    x = np.cos((2 * indices + 1) * np.pi / (2*n))
    #transforming the nodes
    return 0.5 * (start + stop) + 0.5 * (stop - start) * x

def equidistant_nodes(n, start, stop):
    return np.linspace(start, stop, n)

def subdivide(start, stop, n_intervals):
    length = stop - start
    interval_length = length / n_intervals
    intervals = []

    for i in range(n_intervals):
        #Calculating lower (a) and upper (b) boundary for each sub-interval
        a = start + i * interval_length
        b = a + interval_length
        intervals.append((a, b))    #Appending the boundaries as a coordinate pair

    return intervals

def piecewise_equidistant_lagrange(f, start, stop, n_intervals, num_nodes, num_x):
    intervals = subdivide(start, stop, n_intervals) #Constructing sub -ntervals
    lagranges = []
    for interval in intervals:
        interval_start = interval[0]
        interval_stop = interval[-1]

        x = np.linspace(interval_start, interval_stop, num_x)   #Constructing the "real line" in each sub-interval

        #Constructing nodes and corresponding values
        nodes = equidistant_nodes(num_nodes, interval_start, interval_stop)
        vals = f(nodes)

        p = lagrange_interpolation_polynomial(x, nodes, vals)   #Computing the lagrange interpolation approximation for each sub-interval
        lagranges.append((p, x))
    return lagranges    #Returning the approximation and its domain for all sub-intervals