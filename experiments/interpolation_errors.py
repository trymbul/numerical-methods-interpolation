import numpy as np

from src.functions import (
    cos_func,
    e_func,
)

from src.interpolation import (
    equidistant_nodes,
    chebishev_nodes,
    lagrange_interpolation_polynomial,
)

from src.error_analysis import (
    max_norm_error,
    two_norm_error,
    error_bound_cos2pi,
)

from src.plotting import (
    plot_error_curves,
    plot_error_bound_comparison,
    save_table,
)

def compute_cos_error_analysis():

    import time

    max_degree = 25

    num_x = 100 * max_degree

    global_degrees = np.arange(
        1,
        max_degree + 1,
    )

    start = 0
    stop = 1

    x = np.linspace(
        start,
        stop,
        num_x,
    )

    eq_2_error = np.zeros(max_degree)
    ch_2_error = np.zeros(max_degree)

    eq_max_error = np.zeros(max_degree)
    ch_max_error = np.zeros(max_degree)

    eq_error_bound = np.zeros(max_degree)
    ch_error_bound = np.zeros(max_degree)

    global_eq_timings = np.zeros(max_degree)
    global_ch_timings = np.zeros(max_degree)

    for degree in global_degrees:

        num_nodes = degree + 1

        i = degree - 1

        start_time_eq = time.perf_counter()

        # Equidistant nodes
        eq_nodes = equidistant_nodes(
            num_nodes,
            start,
            stop,
        )

        eq_vals = cos_func(eq_nodes)

        p_eq = lagrange_interpolation_polynomial(
            x,
            eq_nodes,
            eq_vals,
        )

        eq_2_error[i] = two_norm_error(
            cos_func,
            p_eq,
            x,
            start,
            stop,
        )

        eq_max_error[i] = max_norm_error(
            cos_func,
            p_eq,
            x,
        )

        eq_error_bound[i] = error_bound_cos2pi(
            degree,
            eq_nodes,
            x,
        )

        global_eq_timings[i] = (
            time.perf_counter()
            - start_time_eq
        )

        # Chebyshev nodes
        start_time_ch = time.perf_counter()

        ch_nodes = chebishev_nodes(
            num_nodes,
            start,
            stop,
        )

        ch_vals = cos_func(ch_nodes)

        p_ch = lagrange_interpolation_polynomial(
            x,
            ch_nodes,
            ch_vals,
        )

        ch_2_error[i] = two_norm_error(
            cos_func,
            p_ch,
            x,
            start,
            stop,
        )

        ch_max_error[i] = max_norm_error(
            cos_func,
            p_ch,
            x,
        )

        ch_error_bound[i] = error_bound_cos2pi(
            degree,
            ch_nodes,
            x,
        )

        global_ch_timings[i] = (
            time.perf_counter()
            - start_time_ch
        )

    return (
        global_degrees,
        eq_2_error,
        eq_max_error,
        ch_2_error,
        ch_max_error,
        eq_error_bound,
        ch_error_bound,
        global_eq_timings,
        global_ch_timings,
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


plot_error_bound_comparison(
    global_degrees,
    eq_max_error,
    eq_2_error,
    ch_max_error,
    ch_2_error,
    eq_error_bound,
    ch_error_bound,
    "cos_error_bounds.png",
)

global_headers = [
    "Method",
    "Nodes",
    "Max Error",
    "Time [ms]",
]

global_rows = []

for i in [9, 19, 24]:

    eq_err = eq_max_error[i]
    eq_time = global_eq_timings[i] * 1000

    global_rows.append([
        "Equidistant",
        i + 1,
        f"{eq_err:.2e}",
        f"{eq_time:.2f}",
    ])

    ch_err = ch_max_error[i]
    ch_time = global_ch_timings[i] * 1000

    global_rows.append([
        "Chebyshev",
        i + 1,
        f"{ch_err:.2e}",
        f"{ch_time:.2f}",
    ])

save_table(
    global_headers,
    global_rows,
    "global_interpolation_table.png",
)

#Plot for e^(3x) * sin(2x)
start = 0
stop = np.pi/4
max_degree = 25
num_x = 100 * max_degree

x = np.linspace(start, stop, num_x)

#Initialising the error arrays
eq_2_error_e = np.zeros(max_degree)
ch_2_error_e = np.zeros(max_degree)

eq_max_error_e = np.zeros(max_degree)
ch_max_error_e = np.zeros(max_degree)

#Computing the error of the approximation
for degree in global_degrees:

    num_nodes = degree + 1

    #Constructing nodes and corresponding values
    eq_nodes_e = equidistant_nodes(num_nodes, start, stop)
    ch_nodes_e = chebishev_nodes(num_nodes, start, stop)

    eq_vals_e = e_func(eq_nodes_e)
    ch_vals_e = e_func(ch_nodes_e)

    #Computing Lagrange interpolation polynomiial both sets of nodes
    p_eq_e = lagrange_interpolation_polynomial(x, eq_nodes_e, eq_vals_e)
    p_ch_e = lagrange_interpolation_polynomial(x, ch_nodes_e, ch_vals_e)

    #Computing the errors for both norms and both sets of nodes
    i = degree-1
    eq_2_error_e[i] = two_norm_error(e_func, p_eq_e, x, start, stop)
    ch_2_error_e[i] = two_norm_error(e_func, p_ch_e, x, start, stop)

    eq_max_error_e[i] = max_norm_error(e_func, p_eq_e, x)
    ch_max_error_e[i] = max_norm_error(e_func, p_ch_e, x)

plot_error_curves(
    global_degrees,
    eq_2_error_e,
    eq_max_error_e,
    ch_2_error_e,
    ch_max_error_e,
    r"Interpolation errors for $e^{3x}\sin(2x)$",
    "exp_sin_interpolation_errors.png",
)