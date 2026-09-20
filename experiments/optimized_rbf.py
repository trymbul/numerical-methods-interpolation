import autograd.numpy as np
from src.functions import Runge_func

from src.interpolation import (
    chebishev_nodes,
)

from src.rbf import RBF

from src.error_analysis import (
    two_norm_error,
)

from src.optimization import (
    gradient_descent,
)

from src.plotting import (
    plot_gradient_descent_history,
    save_table,
)

#Initial values
start = -5.0
stop = 5.0
N = 1000
degrees = [4, 6, 8, 10, 15, 20]
eps = 1

x = np.linspace(start, stop, N + 1)

headers = [
    "Degree",
    "Eq Error",
    "Ch Error",
    "Opt Error",
    "Opt ε",
]

rows = []
histories = []
labels = []

for degree in degrees:
    num_nodes = degree + 1

    #Equidistant nodes and corresponding values
    eq_nodes = np.linspace(start, stop, num_nodes)
    eq_vals = Runge_func(eq_nodes)
    #RBF approximation for equidistant nodes, and its 2-norm error
    RBF_approx_eq = RBF(eq_nodes, eq_vals, x, eps)
    eq_err = two_norm_error(Runge_func, RBF_approx_eq, x, start, stop)
    
    #Chebishev nodes and corresponding values
    ch_nodes = chebishev_nodes(num_nodes, start, stop)
    ch_vals = Runge_func(ch_nodes)
    #RBF approximation for Chebishev nodes, and its 2-norm error
    RBF_approx_ch = RBF(ch_nodes, ch_vals, x, eps)
    ch_err = two_norm_error(Runge_func, RBF_approx_ch, x, start, stop)
    
    #Finding optimal nodes and epsilon, use it to find the error with optimal values
    opt_nodes, opt_eps, history = gradient_descent(num_nodes, start, stop, N, Runge_func, RBF)
    opt_vals = Runge_func(opt_nodes)
    RBF_approx_opt = RBF(opt_nodes, opt_vals, x, opt_eps)
    opt_err = two_norm_error(Runge_func, RBF_approx_opt, x, start, stop)

    rows.append([
    degree,
    f"{eq_err:.4e}",
    f"{ch_err:.4e}",
    f"{opt_err:.4e}",
    f"{opt_eps:.3f}",
    ])

    histories.append(history)

    labels.append(f"degree={degree}")

save_table(
    headers,
    rows,
    "optimized_rbf_table.png",
)

plot_gradient_descent_history(
    histories,
    labels,
)