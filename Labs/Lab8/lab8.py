import mypkg
import numpy as np
import math
from numpy.linalg import inv 
import matplotlib.pyplot as plt
from numpy.linalg import norm

def driver():
    
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    for i in range(2,11):
        #Nint = i
        yeval = eval_lin_spline(xeval,Neval, a,b,f,i)
    #''' evaluate f at the evaluation points'''
        fex =f(xeval)
        #for j in range(Neval):
            #fex[j] = f(xeval[j]) 
      
        plt.figure()
        plt.plot(xeval,fex,'ro-', label='function')
        plt.plot(xeval,yeval,'bs-',label='linear spline approx')
        plt.legend()
        plt.show()
        
        err = abs(yeval-fex)
        plt.figure()
        plt.plot(xeval,err,'ro-')
        plt.show()    

def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        
        for kk in range(Neval):
            yeval[kk] = eval_lin(a1,fa1, b1, fb1,xeval[kk])
           
    return yeval

def eval_lin(x0, fx0, x1, fx1, alpha):
    m = (fx1 - fx0)/(x1-x0)
    b = fx0 - m * x0
    return m * alpha + b

driver()