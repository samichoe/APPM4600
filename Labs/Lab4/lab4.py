import numpy as np

def driver():
# test functions
    f1 = lambda x: (10/(x+4))**(1/2)
# fixed point is alpha1 = 1.4987....
    f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09...
    Nmax = 100
    tol = 1e-10
# test f1 '''
    x0 = 1.5
    [xstar,ier,count,p] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print(f'num iterations is {count}')
    print('Error message reads:',ier)
    print(f'Alpha is {alpha(p,xstar)}')
#test f2 '''
    x0 = 0.0
    [xstar,ier,count,p] = fixedpt(f2,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f2(xstar):',f2(xstar))
    print('Error message reads:',ier)
# define routines

def fixedpt(f,x0,tol,Nmax):
    #''' x0 = initial guess'''
    #''' Nmax = max number of iterations'''
    #''' tol = stopping tolerance'''
    count = 0
    p = []
    while (count<Nmax):
        p.append(x0)
        count = count +1
        x1 = f(x0)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier,count,p]
        x0 = x1
    xstar = x1
    
    
    ier = 1
    return [xstar, ier,count,p]

def alpha(p,xstar):
    pn1 = p[-1]
    pn = p[-2]
    pnm1 = p[-3]
    alpha = np.log(abs(pn1-xstar)/abs(pn-xstar))/(np.log(abs(pn-xstar)/abs(pnm1-xstar)))
    return alpha

driver()

