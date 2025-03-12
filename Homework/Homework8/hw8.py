import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
from numpy.linalg import norm

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

def eval_hermite(xeval,xint,yint,ypint,N):

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    lpj = np.zeros(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              
    yeval = 0
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)

def create_natural_spline(yint,xint,N):
    # got the psuedo code from https://sites.millersville.edu/rbuchanan/math375/CubicSpline.pdf
    alpha = np.zeros(N)
    h = np.zeros(N)
    h[0] = xint[1]-xint[0]
    
    for i in range(1,N):
       h[i] = xint[i+1] - xint[i]
       alpha[i] = 3/h[i]*(yint[i+1]-yint[i]) - (3/h[i-1])*(yint[i]-yint[i-1])

    l = np.zeros(N+1)
    mu = np.zeros(N)
    z = np.zeros(N+1)
    
    l[0]=1
    mu[0]=0
    z[0] = 0
    for i in np.arange(N-1):
        l[i] = 2*(xint[i+1]-xint[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
    
    l[N] = 1
    B = np.zeros(N)
    C = np.zeros(N+1)
    z[N] =0 
    A = np.zeros(N+1)
    
    for j in range(N - 1, -1, -1):
        C[j] = z[j] - mu[j] *C[j + 1]
        B[j] = (yint[j + 1] - yint[j]) / h[j] - h[j] * (C[j + 1] + 2 * C[j]) / 3
        A[j] = yint[j]
    return A, B, C
    

def create_clamped_spline(yint, xint, N, fp0, fpN):
    # got the psuedo code from https://sites.millersville.edu/rbuchanan/math375/CubicSpline.pdf
    # yint = a[i]
    
    alpha = np.zeros(N+1)
    h = np.zeros(N)
    h[0] = xint[1] - xint[0]
    B = np.zeros(N)
    C = np.zeros(N+1)
    
    for i in range(1, N):
        h[i] = xint[i + 1] - xint[i]
        alpha[i] = (3/h[i])*(yint[i + 1] - yint[i]) - (3/h[i])*(yint[i] - yint[i - 1])
    alpha[0] = 3*(yint[1] - yint[0])/(h[0] - 3*fp0)
    alpha[N] = 3*fpN - 3*(yint[N]-yint[N-1])/h[N-1]
    
    l=np.zeros(N +1)
    mu=np.zeros(N)
    z = np.zeros(N + 1)
    
    l[0] = 2 * h[0]
    mu[0] =0.5
    z[0] = alpha[0]/l[0]
    
    for i in range(1, N):
       l[i] = 2*(xint[i+1]-xint[i-1]) - h[i-1]*mu[i-1]
       mu[i] = h[i]/l[i]
       z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i]
    
    l[N] = h[N-1]*(2-mu[N-1])
    z[N] = (alpha[N]-h[N-1]*z[N-1])/l[N]
    C[N]= z[N]
    A = np.zeros(N+1)
    
    for j in range(N - 1, -1, -1):
        C[j] = z[j] - mu[j] *C[j + 1]
        B[j] = (yint[j + 1] - yint[j]) / h[j] - h[j] * (C[j + 1] + 2 * C[j]) / 3
        A[j] = yint[j]
    return A, B, C


def  eval_cubic_spline(xeval,Neval,xint,Nint,A,B,C):

    yeval = np.zeros(Neval+1);

    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j];
        btmp= xint[j+1];

    #   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp));
        xloc = xeval[ind];

    # evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,A[j],A[j+1],B[j],C[j])
    #   copy into yeval
        yeval[ind] = yloc

    return(yeval)

def eval_local_spline(xeval,xi,xip,Ai,Aip,B,C):
    hi = xip-xi
    yeval = Ai + B * (xeval - xi) + C * (xeval - xi) ** 2 + ((Aip - Ai - B * hi - C * hi ** 2) / hi ** 3) * (xeval - xi) ** 3
    
    return yeval

## Problem 1
def interp(n,err=False):
    f = lambda x: 1/(1+x**2)
    fp = lambda x: -2*x/(1+x**2)**2
    a = -5
    b = 5
    fp0 = fp(a)
    fpN = fp(b)
    
    for N in n:
        xint = np.linspace(a,b,N+1)

        yint = f(xint)
        yint_H = np.zeros(N+1)
        
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        
        yeval_l = np.zeros(Neval+1)
        yeval_H = np.zeros(Neval+1)
        (A,B,C) = create_natural_spline(yint,xint,N)
        yeval_ns = eval_cubic_spline(xeval,Neval,xint,N,A,B,C)
        (A,B,C) = create_clamped_spline(yint,xint,N,fp0,fpN)
        yeval_cs = eval_cubic_spline(xeval,Neval,xint,N,A,B,C)
        y = np.zeros( (N+1, N+1) )
     
        for j in range(N+1):
            y[j][0]  = yint[j]
            yint_H[j] = fp(xint[j])

        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
            yeval_H[kk] = eval_hermite(xeval[kk],xint,yint,yint_H,N)

        fex = f(xeval)
       

        plt.figure()    
        plt.plot(xeval,fex, label= "f(x)")
        plt.plot(xeval,yeval_l,ls='--',label="Lagrange")
        plt.plot(xeval,yeval_H,ls='--',label='Hermite')
        plt.plot(xeval,yeval_ns,ls='--',label='Natural spline')
        plt.plot(xeval,yeval_cs,ls='--',label='Clamped spline') 
        plt.title(f'Interpolation Comparisons, n = {N}')
        plt.legend()
        plt.show()
        
        if err:
            err_l = abs(yeval_l-fex)
            err_H = abs(yeval_H-fex)
            err_ns = abs(yeval_ns-fex)
            err_cs = abs(yeval_cs-fex)
            plt.figure()
            plt.semilogy(xeval,err_l,'--',label='Lagrange')
            plt.semilogy(xeval,err_H,'--',label='Hermite')
            plt.semilogy(xeval,err_ns,'--',label='Natural Spline')
            plt.semilogy(xeval,err_cs,'--',label='Cubic Spline')
            plt.title(f'Error Plots, n={N}')
            plt.legend()
            plt.show()            
        
n = [5, 10, 15, 20]
interp(n)

## Problem 2
def cheb(n,err=False):
    f = lambda x: 1/(1+x**2)
    fp = lambda x: -2*x/(1+x**2)**2
    a = -5
    b = 5
    fp0 = fp(a)
    fpN = fp(b)
    
    for N in n:
        xint = np.array([ (a + b) / 2 + (b - a) / 2 * np.cos((2 * j - 1) * np.pi / (2 * (N + 1))) for j in range(1, N + 2) ])
        yint = f(xint)
        yint_H = np.zeros(N+1)
        
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        
        yeval_l = np.zeros(Neval+1)
        yeval_H = np.zeros(Neval+1)
        (A,B,C) = create_natural_spline(yint,xint,N)
        yeval_ns = eval_cubic_spline(xeval,Neval,xint,N,A,B,C)
        (A,B,C) = create_clamped_spline(yint,xint,N,fp0,fpN)
        yeval_cs = eval_cubic_spline(xeval,Neval,xint,N,A,B,C)
        y = np.zeros( (N+1, N+1) )
     
        for j in range(N+1):
            y[j][0]  = yint[j]
            yint_H[j] = fp(xint[j])

        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
            yeval_H[kk] = eval_hermite(xeval[kk],xint,yint,yint_H,N)

        fex = f(xeval)
       

        plt.figure()    
        plt.plot(xeval,fex, label= "f(x)")
        plt.plot(xeval,yeval_l,ls='--',label="Lagrange")
        plt.plot(xeval,yeval_H,ls='--',label='Hermite')
        plt.plot(xeval,yeval_ns,ls='--',label='Natural spline')
        plt.plot(xeval,yeval_cs,ls='--',label='Clamped spline') 
        plt.title(f'Interpolation Comparisons, Chebyshev n = {N}')
        plt.legend()
        plt.show()
        
        if err:
            err_l = abs(yeval_l-fex)
            err_H = abs(yeval_H-fex)
            err_ns = abs(yeval_ns-fex)
            err_cs = abs(yeval_cs-fex)
            plt.figure()
            plt.semilogy(xeval,err_l,'--',label='Lagrange')
            plt.semilogy(xeval,err_H,'--',label='Hermite')
            plt.semilogy(xeval,err_ns,'--',label='Natural Spline')
            lt.semilogy(xeval,err_cs,'--',label='Cubic Spline')
            plt.title(f'Error Plots, Chebyshev n={N}')
            plt.legend()
            plt.show()            
        
cheb(n)

## Problem 3
def create_periodic_spline(yint,xint,N):
    b = np.zeros(N-1)
    h = np.zeros(N)
    h[0] = xint[1]-xint[0]
    
    for i in range(1,N):
       h[i] = xint[i+1] - xint[i]
       b[i-1] = ((yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1])/(h[i-1]+h[i]);
    
    M = np.zeros((N-1,N-1))
    
    for i in np.arange(N-1):
        M[i,i] = 4/12;

        if i<(N-2):
            M[i,i+1] = h[i+1]/(6*(h[i]+h[i+1]));

        if i>0:
            M[i,i-1] = h[i]/(6*(h[i]+h[i+1]));
    M[0,N-2] = 2*h[1]/(h[1]+h[2])
    M[N-2,0] = 2*h[N-1] /(h[N-2]+h[N-1])
    
# Solve system M*A = b to find coefficients (a[1],a[2],...,a[N-1]).
    A = np.zeros(N+1);
    A[1:N] = np.linalg.solve(M, b)
    A[0] = A[N]
#  Create the linear coefficients
    B = np.zeros(N)
    C = np.zeros(N)
    for j in range(N):
       B[j] =(yint[j+1] - yint[j]) / h[j] - (A[j+1] + 2 * A[j]) * h[j] / 6
       C[j] = (A[j+1] - A[j]) / (6 * h[j])
    
    B[0] = B[N-1]
    C[0] = C[N-1]
    return(A,B,C)


def per(n):
    f = lambda x: np.sin(10*x)
    
    a = 0
    b = 2*np.pi
    Nint=n
    xint = np.linspace(a,b,Nint+1);
    yint = f(xint);

   
    Neval = 1000;
    xeval =  np.linspace(xint[0],xint[Nint],Neval+1);

#   Create the coefficients for the natural spline
    (A,B,C) = create_periodic_spline(yint,xint,Nint);

#  evaluate the cubic spline
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,A,B,C);


    ''' evaluate f at the evaluation points'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,label='Exact function')
    plt.plot(xeval,yeval,label='Periodic cubic spline')
    plt.title(f'N = {Nint}')
    plt.legend(loc="lower left")
    plt.show()
    
per(3)