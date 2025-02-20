# Homework 5 code

import numpy as np
import matplotlib.pyplot as plt
## Problem 1
def f(x,y):
    return 3*x**2 - y**2

def g(x,y):
    return 3*x*y**2 - x**3 - 1

## part a), given 100 iterations
def iter(x0, y0, nmax):
    matrix = np.array([[1/6, 1/18], [0, 1/6]])
    x = x0
    y = y0
    xn= []
    for i in range(nmax):
        xn.append((x,y))
        x, y = np.array([x,y]) - np.dot(matrix, np.array([f(x,y), g(x,y)]))
        print(f'x = {x:.4f}      y = {y:.4f}')
    return np.array(xn), (x,y)

xn , r = iter(1,1,50)

print(f'Iteration system given for part a): x = {r[0]}, y = {r[1]} \nNumber of iterations = 50')

num_iter = xn.shape[0]
err = np.max(np.abs(xn - r), 1)
plt.plot(np.arange(num_iter), np.log10(err + 1e-18), 'b-o', label='Iteration Scheme')
plt.title('Iteration Scheme log10|r - rn|')
plt.legend()
plt.show()

## part c) Newton's method
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);


def F(x):
    return np.array([3*x[0]**2-x[1]**2 , 3*x[1]**2*x[0]-x[0]**3-1]);
def JF(x):
    return np.array([[6*x[0],-2*x[1]],[3*x[1]**2-3*x[0]**2,6*x[0]*x[1]]]);
x0 = np.array([1.0,1.0]); tol=1e-10; nmax=100;

(rN,rnN,nfN,nJN) = newton_method_nd(F,JF,x0,tol,nmax,verb=True)
print(f'Newton method x= {rN[0]}, y = {rN[1]}\nNumber of iterations = {nfN}')
numN = rnN.shape[0];
errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
plt.title('Newton iteration log10|r-rn|');
plt.legend();
plt.show();
print(np.sqrt(3)/2)

# Problem 3 b)

def iter2(x0, y0, z0, tol=1e-10, nmax=100):
    x, y, z = x0, y0, z0
    errors = []
    print("f(xn,yn,zn)")
    for i in range(nmax):
        f = x**2 + 4*y**2 + 4*z**2 - 16
        fx = 2*x
        fy= 8*y
        fz= 8*z
        
        denom = fx**2 + fy**2 + fz**2
        if denom == 0:
            break
        
        d = f / denom
        xn= x - d * fx
        yn = y - d * fy
        zn = z - d * fz
        fn = xn**2 + 4*yn**2 + 4*zn**2 - 16
        print(fn)
        error = np.sqrt((xn- x)**2 + (yn - y)**2 + (zn - z)**2)
        errors.append(error)
        
        if error < tol:
            break
        
        x, y, z = xn, yn, zn
    
    return x, y, z, errors

x,y,z,err = iter2(1,1,1)

print(f'Point on ellipsoid: ({x:.4f}, {y:.4f}, {z:.4f})')
