import numpy as np
from src.functions import cos_func

from src.interpolation import piecewise_equidistant_lagrange

from src.error_analysis import piecewice_max_norm_error

from src.plotting import plot_piecewise_convergence, save_table, plot_piecewise_vs_global

from experiments.interpolation_errors import compute_cos_error_analysis



import time

#Initial values
start = 0
stop = 1
max_degree = 10
K_max = 201 #Maximum number of sub intervals
num_x=1000

K_list = np.arange(1, K_max+1) #List of number of sub intervals
errors = [] #List of errors for every degree
total_nodes_list = []
timings = []    #To save computational cost


for degree in range(1, max_degree+1):
    errors.append([])
    total_nodes_list.append([])
    timings.append([])
    num_nodes = degree + 1

    #Finding the max-norm error of the cos(2pix) function for each number of sub-interval
    for K in K_list:
        t0 = time.perf_counter()

        lagranges = piecewise_equidistant_lagrange(cos_func, start, stop, K, num_nodes, num_x)
        errors[degree-1].append((piecewice_max_norm_error(cos_func, lagranges)))

        t1 = time.perf_counter()

        timings[degree-1].append(t1 - t0)
        total_nodes_list[degree-1].append(K * degree + 1)

plot_piecewise_convergence(
    K_list,
    errors,
)

(
    global_degrees,
    eq_2_error,
    eq_max_error,
    ch_2_error,
    ch_max_error,
    eq_error_bound,
    ch_error_bound,
    global_eq_timings,
    global_ch_timings,
) = compute_cos_error_analysis()

plot_piecewise_vs_global(
    total_nodes_list,
    errors,
    global_degrees,
    eq_max_error,
    ch_max_error,
)

max_total_nodes = 201


for i in range(len(errors)):
    degree = i+1
    nodes_per_interval = degree +1 
    intervals = 1
    while (total_nodes_list[i][intervals-1] <= max_total_nodes):
        intervals += 1
    timings[i] = timings[i][:intervals]

headers = [
    "Degree",
    "Intervals",
    "Total Nodes",
    "Max Error",
    "Time [ms]",
]

rows = []

#Degrees for plotting
plot_degrees = [5, 10]
plot_nodes = [1, 11, 51, 81, 101, 151, 201]

for i in range(len(errors)):
    degree = i + 1
    
    #We only print for 5 and 10
    if degree in plot_degrees:
        did_print = False
        
        #Going through all total nodes
        for j in range(len(total_nodes_list[i])):
            nodes = total_nodes_list[i][j]
            
            #Check if the number of nodes is what we are looking for
            if nodes in plot_nodes:
                err = errors[i][j]
                time = timings[i][j] * 1000  #Converting to ms
                num_int = j + 1

                rows.append([degree, num_int, nodes, f"{err:.2e}", f"{time:.2f}",])

save_table(
    headers,
    rows,
    "piecewise_summary.png",
)