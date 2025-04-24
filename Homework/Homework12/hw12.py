import numpy as np

def hilbert_matrix(n):
    return np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])

def power_method(A, tol=1e-10, max_iter=1000):
    n = A.shape[0]
    x = np.ones(n)
    x = x / np.linalg.norm(x)
    
    lambda_guess=0

    for k in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)
        lambda_new = x_new.T @ A @ x_new

        if np.linalg.norm(x_new - x) < tol:
            return lambda_new, x_new, k + 1
        x = x_new
        lambda_guess = lambda_new

    return lambda_new, x_new, max_iter

print('Problem 3(a)')
for n in range(4, 21, 4):
    A = hilbert_matrix(n)
    eival, eivec, niter = power_method(A)
    print(f"n = {n}: lambda = {eival:.6f}, iterations = {niter}")
    
print('\nProblem 3(b)')


def inverse_power_method(A, tol=1e-10, max_iter=1000):
    n = A.shape[0]
    x = np.ones(n)
    x = x / np.linalg.norm(x)

    lambda_guess = 0

    for k in range(max_iter):
        y = np.linalg.solve(A, x)
        x_new = y / np.linalg.norm(y)
        lambda_new = (x_new.T @ A @ x_new) / (x_new.T @ x_new)

        if abs(lambda_new - lambda_guess) < tol:
            return lambda_new, x_new, k + 1

        x = x_new
        lambda_guess = lambda_new

    return lambda_new, x_new, max_iter

A = hilbert_matrix(16)
smallest, eivec, iters = inverse_power_method(A)
print(f"Smallest eigenvalue (n = 16): lambda = {smallest:.4e}, iterations = {iters}")

def acc(A,l, eivec):
    Ax = A @ eivec
    lambda_x = l * eivec
    res = np.linalg.norm(Ax - lambda_x)
    return res

err = acc(A,smallest,eivec)
print(f'The error for calculating the eigenvalue is: {err:.4e}\n')

print('Problem 3(d)')

A = np.array([[0, -1], [1, 0]])
eival, eivec, iters = power_method(A)
print(f"Result after {iters} iterations: lambda = {eival}, x = {eivec}")