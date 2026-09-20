import numpy as np

from src.functions import (
    Runge_func,
    cos_func,
)

from src.interpolation import (
    equidistant_nodes,
)

from src.rbf import (
    RBF,
    M_condition_number,
)

from src.plotting import (
    plot_rbf_approximation,
    plot_rbf_condition_number,
)

#Runge function

#Initial values
start = -5
stop = 5
degree = 10
num_x = 1000
eps = 1

#Constucting equidistant nodes and values
num_nodes = degree + 1
nodes = equidistant_nodes(num_nodes, start, stop)
vals = Runge_func(nodes)

#Computing the exact solution
x = np.linspace(start, stop, num_x)
y = Runge_func(x)

#Computing the RBF interpolation approximation
RBF_approx = RBF(nodes, vals, x, eps)

plot_rbf_approximation(
    x,
    y,
    RBF_approx,
    nodes,
    vals,
    "Runge function and RBF approximation",
    "rbf_runge_function.png",
)

#Cond(M) versus eps for Runge

#Constructing an array of epsilons ranging from 0.001 to 10
eps_values = np.logspace(-3, 1, 200)

#Calculating the condition number for each epsilon
condition_numbers = []
for eps in eps_values:
    condition_numbers.append(M_condition_number(nodes, eps))


#Cond(M) versus eps for Runge

#Constructing an array of epsilons ranging from 0.001 to 10
eps_values = np.logspace(-3, 1, 200)

#Calculating the condition number for each epsilon
condition_numbers = []
for eps in eps_values:
    condition_numbers.append(M_condition_number(nodes, eps))

plot_rbf_condition_number(
    eps_values,
    condition_numbers,
)

#cos(2pi*x)

#Initial values
start = 0
stop = 1
degree = 10
num_x = 1000
eps = 1

#Constructing equidistant nodes and values
num_nodes = degree + 1
nodes = equidistant_nodes(num_nodes, start, stop)
vals = cos_func(nodes)

#Computing the exact solution
x = np.linspace(start, stop, num_x)
y = cos_func(x)

#Computing the RBF interpolation approximation
RBF_approx = RBF(nodes, vals, x, eps)

plot_rbf_approximation(
    x,
    y,
    RBF_approx,
    nodes,
    vals,
    r"$\cos(2\pi x)$ and RBF approximation",
    "rbf_cos_function.png",
)