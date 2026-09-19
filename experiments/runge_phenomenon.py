import numpy as np
import matplotlib.pyplot as plt

from src.functions import Runge_func

from src.interpolation import equidistant_nodes, chebishev_nodes, lagrange_interpolation_polynomial

from src.plotting import plot_interpolation

#Initial values
degree = 10
start = -5
stop = 5
num_x = 10000

#Constructing the exact solution
x = np.linspace(start, stop, num_x)
y = Runge_func(x)

#Constructing the nodes
num_nodes = degree+1    #The number of nodes required for polynomial interpolation of degree n
eq_nodes = equidistant_nodes(num_nodes, start, stop)
ch_nodes = chebishev_nodes(num_nodes, start, stop)

#Constructing the values corresponding to the nodes
eq_vals = Runge_func(eq_nodes)
ch_vals = Runge_func(ch_nodes)

#Approximating the Runge function using the known nodes
lagrange_eq = lagrange_interpolation_polynomial(x, eq_nodes, eq_vals)
lagrange_ch = lagrange_interpolation_polynomial(x, ch_nodes, ch_vals)

plot_interpolation(
    x,
    y,
    lagrange_eq,
    lagrange_ch,
    eq_nodes,
    ch_nodes,
    eq_vals,
    ch_vals,
    "Runge function and Lagrange interpolation",
    "runge_interpolation_degree_10.png",
)


#Initial values
degrees = [15, 20, 30]
start = -5
stop = 5
num_x = 10000

#Constructing the exact solution
x = np.linspace(start, stop, num_x)
y = Runge_func(x)

#Creating a plot for each degree
for degree in degrees:
    num_nodes = degree+1

    #Constructing the nodes and their corresponding values
    eq_nodes = equidistant_nodes(num_nodes, start, stop)
    ch_nodes = chebishev_nodes(num_nodes, start, stop)

    eq_vals = Runge_func(eq_nodes)
    ch_vals = Runge_func(ch_nodes)

    #Approximating the Runge function for each set of nodes
    lagrange_eq = lagrange_interpolation_polynomial(x, eq_nodes, eq_vals)
    lagrange_ch = lagrange_interpolation_polynomial(x, ch_nodes, ch_vals)

    plot_interpolation(
    x,
    y,
    lagrange_eq,
    lagrange_ch,
    eq_nodes,
    ch_nodes,
    eq_vals,
    ch_vals,
    f"Runge interpolation (degree={degree})",
    f"runge_interpolation_degree_{degree}.png",
)


#Initial values
degrees = [10, 15, 20, 30]
start = -1
stop = 1
num_x = 10000

#Creating the excact solution
x = np.linspace(start, stop, num_x)
y = Runge_func(x)

#Plotting for every degree in degrees
for degree in degrees:
    #Constructing nodes and corresponding values
    num_nodes = degree+1
    eq_nodes = equidistant_nodes(num_nodes, start, stop)
    ch_nodes = chebishev_nodes(num_nodes, start, stop)

    eq_vals = Runge_func(eq_nodes)
    ch_vals = Runge_func(ch_nodes)

    #Approximating the Runge function for each set of nodes
    lagrange_eq = lagrange_interpolation_polynomial(x, eq_nodes, eq_vals)
    lagrange_ch = lagrange_interpolation_polynomial(x, ch_nodes, ch_vals)
    plot_interpolation(
    x,
    y,
    lagrange_eq,
    lagrange_ch,
    eq_nodes,
    ch_nodes,
    eq_vals,
    ch_vals,
    f"Runge interpolation on [-1,1] (degree={degree})",
    f"runge_interval_minus1_1_degree_{degree}.png",
)