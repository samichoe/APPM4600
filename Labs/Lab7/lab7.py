import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

def monomial(): 

    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    
    xs = []
    ys = []
    for i in range(2,10):
        N = i
        xint = np.linspace(a,b,N+1)
#    print('xint =',xint)
        yint = f(xint)
#    print('yint =',yint)
        V = Vandermonde(xint,N)
#    print('V = ',V)
        Vinv = inv(V)
#    print('Vinv = ' , Vinv)
        coef = Vinv @ yint  
#    print('coef = ', coef)
# No validate the code
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yeval = eval_monomial(xeval,coef,N,Neval)
        xs.append(xeval)
        ys.append(yeval)
        plt.plot(xeval,f(xeval), label = 'f(x) = $1/1+(10x^2)')
        plt.plot(xeval,yeval,label = 'approximation')
        plt.title(f'Degree = {N}')
        plt.legend(loc='upper right')
        plt.show()
# exact function
        yex = f(xeval)
    
        err =  norm(yex-yeval) 
        print('err = ', err)
    
    return

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V     

monomial()

def driver():


    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    
    for i in range(2,10):
        N = i
        xint = np.linspace(a,b,N+1)
    
        yint = f(xint)
    
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yeval_l= np.zeros(Neval+1)
        yeval_dd = np.zeros(Neval+1)
  
    
        y = np.zeros( (N+1, N+1) )
     
        for j in range(N+1):
            y[j][0]  = yint[j]

            y = dividedDiffTable(xint, y, N+1)
    
        for kk in range(Neval+1):
            yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
            yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
          
        fex = f(xeval)
       

        plt.figure()    
        plt.plot(xeval,fex,label='original function')
        plt.plot(xeval,yeval_l,label='Lagrange') 
        plt.plot(xeval,yeval_dd,label = 'Newton DD',ls='--')
        plt.title(f'Degree = {N}')
        plt.legend()

        plt.figure() 
        err_l = abs(yeval_l-fex)
        err_dd = abs(yeval_dd-fex)
        plt.semilogy(xeval,err_l,label='lagrange')
        plt.semilogy(xeval,err_dd,label='Newton DD',ls='--')
        plt.title(f'Degree = {N}')
        plt.legend()
        plt.show()

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
  
    


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

       

driver()        
