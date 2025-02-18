## Lab 6
import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, Video
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

def slacker_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    
    
    Jn = Jf(xn);

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn);

    n=0;
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");
    
    nstep = 1
    
    while npn>tol and n<=nmax:
        nstep += 1
        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        
        if nstep% 5 == 0:
            Jn = Jf(xn)
            lu, piv = lu_factor(Jn)
            
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Slacker Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Slacker Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn);

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn);

    n=0;
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

def F(x):
    return np.array([4*(x[0]**2)+x[1]**2-4 , x[0]+x[1]-np.sin(x[0]-x[1]) ]);


def JF(x):
    return np.array([[8*x[0],2*x[1]],[1-np.cos(x[0]-x[1]),1+np.cos(x[0]-x[1])]]);

x0 = np.array([1.0,0.0]); tol=1e-10; nmax=100;

lazy_newton_method_nd(F,JF,x0,tol,nmax,verb=True)

slacker_newton_method_nd(F,JF,x0,tol,nmax,verb=True)

def F(x):
    return np.array([(x[0]-4)**2+2*(x[1]-2)**2-32 , x[1]*(x[0]-2)-16 ]);
def JF(x):
    return np.array([[2*(x[0]-4),4*(x[1]-2)],[x[1],(x[0]-2)]]);

    # Apply Newton Method:
x0 = np.array([4.0,4.0]); tol=1e-14; nmax=100
lazy_newton_method_nd(F,JF,x0,tol,nmax,True)
slacker_newton_method_nd(F,JF,x0,tol,nmax,verb=True)
