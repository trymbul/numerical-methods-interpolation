import matplotlib.pyplot as plt

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(
        f"figures/{filename}",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_rbf_condition_number(
    eps_values,
    condition_numbers,
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        eps_values,
        condition_numbers,
        linewidth=2,
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("Condition number")

    plt.title(
        "Condition number versus shape parameter"
    )

    plt.grid(True)

    save_figure("rbf_condition_number.png")



def plot_interpolation(
    x,
    exact,
    lagrange_eq,
    lagrange_ch,
    eq_nodes,
    ch_nodes,
    eq_vals,
    ch_vals,
    title,
    filename,
):
    plt.figure(figsize=(8, 5))

    plt.plot(x, exact, "g", label="Exact function")
    plt.plot(x, lagrange_eq, "r--", label="Equidistant nodes")
    plt.plot(x, lagrange_ch, "b--", label="Chebyshev nodes")

    plt.scatter(eq_nodes, eq_vals, color="red")
    plt.scatter(ch_nodes, ch_vals, color="blue")

    plt.title(title)
    plt.grid(True)
    plt.legend()

    save_figure(filename)


def plot_error_curves(
    degrees,
    eq_2_error,
    eq_max_error,
    ch_2_error,
    ch_max_error,
    title,
    filename,
):
    plt.figure(figsize=(8, 5))

    plt.plot(degrees, eq_2_error, "r", label="Equidistant 2-norm")
    plt.plot(degrees, eq_max_error, "b", label="Equidistant max-norm")

    plt.plot(degrees, ch_2_error, "r--", label="Chebyshev 2-norm")
    plt.plot(degrees, ch_max_error, "b--", label="Chebyshev max-norm")

    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Error")
    plt.title(title)

    plt.grid(True)
    plt.legend()

    save_figure(filename)


def plot_error_bound_comparison(
    degrees,
    eq_error,
    ch_error,
    eq_bound,
    ch_bound,
    filename,
):
    plt.figure(figsize=(8, 5))

    plt.plot(degrees, eq_error, label="Equidistant error")
    plt.plot(degrees, ch_error, label="Chebyshev error")

    plt.plot(
        degrees,
        eq_bound,
        "--",
        label="Equidistant bound",
    )

    plt.plot(
        degrees,
        ch_bound,
        "--",
        label="Chebyshev bound",
    )

    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Error")

    plt.grid(True)
    plt.legend()

    save_figure(filename)


def plot_piecewise_convergence(
    K_values,
    errors,
    degree,
):
    plt.figure(figsize=(8, 5))

    plt.plot(K_values, errors)

    plt.yscale("log")
    plt.xlabel("Number of intervals")
    plt.ylabel("Max error")

    plt.title(
        f"Piecewise interpolation convergence (degree={degree})"
    )

    plt.grid(True)

    save_figure("piecewise_convergence.png")



def plot_gradient_descent_history(
    histories,
    labels,
):
    plt.figure(figsize=(8, 5))

    for history, label in zip(histories, labels):
        plt.plot(history, label=label)

    plt.yscale("log")

    plt.xlabel("Iteration")
    plt.ylabel("Cost")

    plt.title("Gradient descent convergence")

    plt.grid(True)
    plt.legend()

    save_figure("rbf_optimization_convergence.png")

def plot_rbf_approximation(
    x,
    exact,
    approximation,
    nodes,
    values,
    title,
    filename,
):
    plt.figure(figsize=(8,5))

    plt.plot(
        x,
        exact,
        "r",
        label="Exact function",
    )

    plt.plot(
        x,
        approximation,
        "b--",
        label="RBF approximation",
    )

    plt.scatter(
        nodes,
        values,
        color="blue",
        label="Interpolation nodes",
    )

    plt.legend()
    plt.grid(True)
    plt.title(title)

    save_figure(filename)