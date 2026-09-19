# Numerical Methods: Interpolation, Approximation and Optimization

A Python project exploring interpolation and approximation techniques from scientific computing and numerical analysis.

The project investigates polynomial interpolation, radial basis function (RBF) interpolation, numerical error analysis and gradient-based optimization. The implementations were developed from scratch to study both theoretical and practical aspects of approximation methods.

## Overview

The project contains implementations and experiments related to:

- Lagrange polynomial interpolation
- Runge phenomenon
- Chebyshev node distributions
- Piecewise polynomial interpolation
- Radial Basis Function (RBF) interpolation
- Error analysis and convergence studies
- Condition number analysis
- Gradient-based optimization of interpolation parameters

The focus is on understanding accuracy, stability and computational trade-offs in numerical approximation methods.

---

## Main Topics

### Polynomial Interpolation

Implemented Lagrange interpolation from scratch and compared different node distributions:

- Equidistant nodes
- Chebyshev nodes

The project demonstrates how node placement affects interpolation accuracy and reproduces the classical Runge phenomenon.

### Error Analysis

Studied approximation error using:

- Maximum norm
- Discrete 2-norm
- Classical interpolation error bounds

Numerical results were compared with theoretical predictions.

### Piecewise Interpolation

Implemented piecewise polynomial interpolation and investigated convergence as the number of subintervals increases.

The experiments illustrate how local approximations can improve accuracy while maintaining numerical stability.

### Radial Basis Function Interpolation

Implemented Gaussian RBF interpolation and analyzed:

- Approximation quality
- Shape-parameter sensitivity
- Conditioning of the interpolation matrix

The project highlights the trade-off between interpolation accuracy and numerical stability.

### Optimization

Used automatic differentiation and gradient descent with backtracking line search to optimize:

- Node locations
- RBF shape parameter ε

The optimized configurations consistently reduced approximation error compared to standard node distributions.

---

## Technologies

- Python
- NumPy
- Matplotlib
- Autograd

---

## Example Results

### Runge Phenomenon

Global polynomial interpolation was tested using both equidistant and Chebyshev nodes. The experiments demonstrate the instability of high-degree interpolation on equidistant nodes and the improved behavior obtained with Chebyshev nodes.

### RBF Conditioning

The condition number of the interpolation matrix was analyzed as a function of the RBF shape parameter.

The results illustrate the well-known trade-off:

- Small ε improves approximation quality
- Small ε also increases ill-conditioning

### Optimized Interpolation Nodes

Gradient-based optimization was used to determine node placements that minimize approximation error directly.

---

## Repository Structure

```text
src/
    Core implementations

figures/
    Generated plots and visualizations

notebooks/
    Interactive experiments

report/
    Full project report