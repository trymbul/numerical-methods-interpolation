import matplotlib.pyplot as plt
import numpy as np

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
    eq_max_error,
    eq_2_error,
    ch_max_error,
    ch_2_error,
    eq_bound,
    ch_bound,
    filename,
):
    plt.figure(figsize=(8, 5))

    plt.plot(degrees, eq_max_error, label="Equidistant max-norm error")
    plt.plot(degrees, eq_2_error, label="Equidistant two-norm error")
    plt.plot(degrees, ch_max_error, label="Chebyshev max-norm error")
    plt.plot(degrees, ch_2_error, label="Chebyshev two-norm error")

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
):
    plt.figure(figsize=(8, 5))

    for i, error_curve in enumerate(errors):

        plt.plot(
            K_values,
            error_curve,
            label=f"degree={i+1}",
        )

    plt.yscale("log")

    plt.xlabel("Number of intervals")
    plt.ylabel("Max error")

    plt.title(
        "Piecewise interpolation convergence"
    )

    plt.grid(True)
    plt.legend()

    save_figure(
        "piecewise_convergence.png"
    )




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

def save_table(
    headers,
    rows,
    filename,
):
    fig, ax = plt.subplots(
        figsize=(8, 3)
    )

    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Header styling
    for (row, col), cell in table.get_celld().items():

        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#D9D9D9")

    fig.tight_layout(pad=0)

    plt.savefig(
        f"figures/{filename}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

def plot_piecewise_vs_global(
    total_nodes_list,
    errors,
    global_degrees,
    eq_max_error,
    ch_max_error,
):
    plt.figure(figsize=(8, 5))

    max_total_nodes = 201

    for i in range(len(errors)):

        intervals = 1

        while (
            total_nodes_list[i][intervals - 1]
            <= max_total_nodes
        ):
            intervals += 1

        else:
            plt.plot(
                total_nodes_list[i][:intervals],
                errors[i][:intervals],
            )

    plt.plot(
        global_degrees,
        eq_max_error,
        "r",
        label="Global equidistant",
    )

    plt.plot(
        global_degrees,
        ch_max_error,
        "b",
        label="Global Chebyshev",
    )

    plt.xlabel("Total interpolation nodes")
    plt.ylabel("Max error")

    plt.yscale("log")

    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title('piecewise vs global')
    
    save_figure('piecewise_vs_global.png')