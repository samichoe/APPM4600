import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():

    f = lambda x: np.sin(x)
    # Taylor
    t = lambda x: x-x**3/6+x**5/120
   # pade rational approx
    R33 = lambda x: (x-(7/60)*x**3)/(1+(1/20)*x**2)
    R24 = lambda x: (x)/(1+(1/6)*x**2+(7/360)*x**4)
    R42 = lambda x: (x-(7/60)*x**3)/(1+(1/20)*x**2)
    # note: r3,3 and r4,2 are the same
    a= 0
    b =5


    # test the different evaluation methods
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
  
    
    # compare the errors
    fex = f(xeval)
    f_rat33 = R33(xeval)
    f_rat24 = R24(xeval)
    f_rat42 = R42(xeval)
    ft = t(xeval)
  
    plt.figure() 
    
    plt.semilogy(xeval,abs(fex-f_rat33),'--',label='Pade, $R_{3,3}$',alpha=0.75)
    plt.semilogy(xeval,abs(fex-f_rat24),':',label='Pade, $R_{2,4}$',alpha=0.75)
    plt.semilogy(xeval,abs(fex-f_rat42),'-.',label='Pade, $R_{4,2}$',alpha=0.75,color='red')
    plt.semilogy(xeval,abs(fex-ft),label='Taylor',color= 'green',alpha=0.5)
    plt.title('Error Plots for Pade Approximations vs Taylor Expansion')
    plt.legend()
    plt.show()
     
driver()
