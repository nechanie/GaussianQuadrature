import matplotlib.pyplot as plt
from mainscript import main
import numpy as np
from scipy.integrate import quad

plt.rcParams.update({'font.size': 16})

def trapezoidal_rule(func, a, b, n):
    h = (b - a) / n
    result = 0.5 * (func(a) + func(b))
    for i in range(1, n):
        result += func(a + i * h)
    return result * h

def simpsons_rule(func, a, b, n):
    h = (b - a) / n
    result = func(a) + func(b)
    for i in range(1, n, 2):
        result += 4 * func(a + i * h)
    for i in range(2, n-1, 2):
        result += 2 * func(a + i * h)
    return result * h / 3

def compare_methods(func, chebyshev_function, exact_integral, interval, max_n):
    gauss_legendre_errors = []
    gauss_chebyshev_errors = []
    trapezoidal_errors = []
    simpsons_errors = []
    n_values = [2**k for k in range(max_n + 1)]

    for n in n_values:
        print(n)
        gauss_legendre_result = main(n, interval, func, 'legendre')[0]
        gauss_chebyshev_result = main(n, interval, chebyshev_function, 'chebyshev')[0]
        trapezoidal_result = trapezoidal_rule(func, interval[0], interval[1], n)
        simpsons_result = simpsons_rule(func, interval[0], interval[1], n)

        gauss_legendre_errors.append(abs(gauss_legendre_result - exact_integral))
        gauss_chebyshev_errors.append(abs(gauss_chebyshev_result - exact_integral))
        trapezoidal_errors.append(abs(trapezoidal_result - exact_integral))
        simpsons_errors.append(abs(simpsons_result - exact_integral))

    plt.scatter(n_values, gauss_legendre_errors, label='Gauss-Legendre', marker='o')
    plt.scatter(n_values, gauss_chebyshev_errors, label='Gauss-Chebyshev', marker='s')
    plt.scatter(n_values, trapezoidal_errors, label='Trapezoidal Rule', marker='^')
    plt.scatter(n_values, simpsons_errors, label='Simpson\'s Rule', marker='D')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Number of Function Evaluations (n)', fontsize=18)
    plt.ylabel('Absolute Error', fontsize=18)
    plt.title('Convergence Comparison', fontsize=20)
    plt.legend()
    plt.show()

def chebyshev_function(x):
    # Example function
    # Leave out the weight function 
    # 1/sqrt(1-x^2), it is already accounted for
    # in the Gaussian Quadrature function
    base_func = np.cos(x)
    return base_func

def standard_function(x):
    # Avoiding the undefined region near -1 and 1
    # This is here to account for the comparison of
    # Gauss-Chebyshev which will be used with functions
    # that will normally include division by zero.
    eps = 1e-6
    x_clipped = np.clip(x, -1 + eps, 1 - eps)

    base_func = np.cos(x_clipped)
    weighted = 1 / np.sqrt(1 - x_clipped**2)
    return base_func * weighted

# Define interval of integration
a, b = -1, 1

# Calculate the approximate integral using SciPy's quad function
exact_integral_value, _ = quad(standard_function, a, b)

# Create visual comparison of various numerical integration methods
compare_methods(standard_function, chebyshev_function, exact_integral_value, (a, b), 7)
