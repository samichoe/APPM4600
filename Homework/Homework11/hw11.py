import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return 1 / (1 + x**2)

def f2(x):
    return (6 * x**2 - 2) / (1 + x**2)**3

def f4(x):
    return (120 * x**4 - 240 * x**2 + 24)/ ((1 + x**2)**5)

def trap(a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    Tn = h/2 * (y[0] + 2* np.sum(y[1:-1]) + y[-1])
    return Tn

def simpsons(a, b, n):
    if n%2 != 0:
        print('n must be even for Simpsons')
        return
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    Sn = h / 3 * (y[0] + 2 * np.sum(y[2:n:2]) + 4 * np.sum(y[1:n:2]) + y[n])
    return Sn

def plotter():
    x = np.linspace(-5,5,100)
    ytrap = f2(x)
    ysim = f4(x)
    plt.plot(x,ytrap,label='second derivative')
    plt.plot(x,ysim,label = 'fourth derivative')
    plt.grid()
    plt.legend()
    plt.show()
    
def maxntrap(tol, a, b):
    max_f2=2
    n = 1 
    while abs(((b - a)**3 / (12 * n**2)) * max_f2) >= tol:
        n += 1
    return n

def maxnsim(tol, a, b):
    max_f4 = 24
    n = 1 
    while abs(((b - a)**5 / (180 * n**4)) * max_f4) >= tol:
        n += 1
    return n

def driver():
    tol4 = 1e-4
    tol6 = 1e-6
    a, b = -5, 5
    
    ntrap4 =maxntrap(tol4,a,b)
    nsim4 =maxnsim(tol4,a,b)
    
    ntrap6 =maxntrap(tol6,a,b)
    nsim6 =maxnsim(tol6,a,b)
    
    res4,err,info4 = quad(f, a, b, full_output=1, epsabs=tol4)[0:3]
    t4 = trap(a,b,ntrap4)
    s4 = simpsons(a,b,nsim4)
    

    res6,err,info6 = quad(f, a, b, full_output=1, epsabs=tol6)[0:3]
    t6 = trap(a,b,ntrap6)
    s6 = simpsons(a,b,nsim6)
    
    print('')
    print(f"{'Method':<12}{'Tolerance':<15}{'Result':<15}{'Function Evals'}")
    print("-" * 60)
    print(f"{'Trapezoidal':<15}{'1e-4':<10}{t4:<20.10f}{ntrap4 + 1}")
    print(f"{'Simpsons':<15}{'1e-4':<10}{s4:<20.10f}{nsim4 + 1}")
    print(f"{'quad':<15}{'1e-4':<10}{res4:<20.10f}{info4['neval']}\n")
    print(f"{'Trapezoidal':<15}{'1e-6':<10}{t6:<20.10f}{ntrap6 + 1}")
    print(f"{'Simpsons':<15}{'1e-6':<10}{s6:<20.10f}{nsim6 + 1}")
    print(f"{'quad':<15}{'1e-6':<10}{res6:<20.10f}{info6['neval']}\n")

plotter()  
driver()

