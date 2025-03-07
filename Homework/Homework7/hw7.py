import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + (10 * x) ** 2)

## Problem 1
def Vmat(x):
    n = len(x)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            V[i, j] = x[i] ** j
    return V

def C(x, y):
    V = Vmat(x)
    V_inv = np.linalg.inv(V)
    c = V_inv @ y
    return c

def p(c, x):
    y = np.zeros_like(x)
    for i in range(len(c)):
        y += c[i] * (x ** i)
    return y

def interp(N):
    h = 2 / (N - 1)
    x_points = np.array([-1 + (i - 1) * h for i in range(1, N + 1)])
    y_points = f(x_points)
    x_interp = np.linspace(-1, 1, 1001)
    
    c = C(x_points, y_points)
    y_poly = p(c, x_interp)
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(x_interp, f(x_interp), label='f(x)', linestyle='dashed')
    plt.plot(x_interp, y_poly, label=f'Interpolating Polynomial (N={N})')
    plt.plot(x_points, y_points, color='red', label='Data Points', marker='o',lw=0)
    plt.ylim(-1, 2)
    plt.legend()
    plt.title(f'Polynomial Interpolation for N={N}')
    plt.show()
    
for N in [2, 3, 5, 10, 15, 17, 18, 19, 20]:
   interp(N)


## Problem 2 (chose the first formula)
def weights(x):
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] /= (x[j] - x[i])
    return w

def lagrange(x, y, w, x_interp):
    n = len(x)
    p = np.ones_like(x_interp)
    phi = np.ones_like(x_interp)
    
    for j in range(n):
        term = w[j] / (x_interp - x[j])
        phi *= (x_interp - x[j])
        p += term * y[j]
    
    p *= phi
    return p

def l_interp(N):
    h = 2 / (N - 1)
    x_points = np.array([-1 + (i - 1) * h for i in range(1, N + 1)])
    y_points = f(x_points)
    x_interp = np.linspace(-1, 1, 1001)
    
    w = weights(x_points)
    y_poly = lagrange(x_points, y_points, w, x_interp)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_interp, f(x_interp), label='f(x)', linestyle='dashed')
    plt.plot(x_interp, y_poly, label=f'Barycentric Interpolation (N={N})')
    plt.plot(x_points, y_points, color='red', label='Data Points', marker='o',lw=0)
    plt.ylim(-1, 2)
    plt.legend()
    plt.title(f'Barycentric Lagrange Interpolation for N={N}')
    plt.show()

for N in [2, 3, 5, 10, 15, 20]:
    l_interp(N)


# Problem 3
def chebyshev(N):
    x_points = np.array([np.cos((2 * j - 1) * np.pi / (2 * N)) for j in range(1, N + 1)])
    y_points = f(x_points)
    x_interp = np.linspace(-1, 1, 1001)
    
    w = weights(x_points)
    y_poly = lagrange(x_points, y_points, w, x_interp)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_interp, f(x_interp), label='f(x)', linestyle='dashed')
    plt.plot(x_interp, y_poly, label=f'Barycentric Interpolation (N={N})')
    plt.scatter(x_points, y_points, color='red', label='Data Points', zorder=3)
    plt.ylim(-1, 2)
    plt.legend()
    plt.title(f'Barycentric Lagrange Interpolation with Chebyshev Points for N={N}')
    plt.show()


for N in [2, 3, 5, 10, 15, 20]:
    chebyshev(N)