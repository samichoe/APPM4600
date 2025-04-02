import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():

    f = lambda x: np.exp(-x)
    # Taylor
    t = lambda x: 1-x+x**2/2-x**3/6+x**4/24-x**5/120
   # pade rational approx
    r = lambda x: (1 - 3/5*x + 3/20*x**2-1/60*x**3)/    \
      (1 + 2/5*x + 1/20*x**2)
      
    a= -0.25
    b =.25
    
    Nint = 3
    # create chebychev nodes
    xint = np.zeros(Nint+1)
    for j in range(1,Nint+2):
       xint[j-1] = np.cos(np.pi*(2*j-1)/(2*(Nint+1)))
    # scale  for the interval
    m = (b-a)/2 
    c = (a+b)/2
    xint = m*xint+c
    xint = xint[::-1]
    
    yint = f(xint)


    # test the different evaluation methods
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
  
    # create the Lagrange evaluation
    yeval = np.zeros(Neval+1) 
    for kk in range(Neval+1):
       yeval[kk] = eval_lagrange(xeval[kk],xint,yint,Nint)       

    # compare the errors
    fex = f(xeval)
    f_rat = r(xeval)
    ft = t(xeval)
  
    plt.figure() 
    
    plt.semilogy(xeval,abs(fex-yeval),'ro--',label='lagrange')
    plt.semilogy(xeval,abs(fex-f_rat),'bs--',label='rational')
    plt.semilogy(xeval,abs(fex-ft),'gs--',label='Taylor')
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
    
    
driver()    
       
