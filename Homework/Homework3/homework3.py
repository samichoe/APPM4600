# import libraries
import numpy as np
import matplotlib.pyplot as plt

def driver():

# use routines 
# homework 3 problem 1 
    f = lambda x: 2*x-1-np.sin(x)
    a = -np.pi
    b = np.pi

    tol = 1e-8

    [astar,ier,count] = bisection(f,a,b,tol)
    print('Homework 1c), f(x) = 2x-1-sinx')
    print('the approximate root is',astar)
    if ier != 0:
      print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'the number of iterations used was {count}')
    print('')
    
    #2a
    f = lambda x:(x-5)**9
    a = 4.82
    b = 5.2
    tol = 1e-4
    [astar,ier,count] = bisection(f,a,b,tol)
    print('Homework 2a), f(x) = (x-5)^9')
    print('the approximate root is',astar)
    if ier != 0:
      print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'the number of iterations used was {count}')
    print('')
    
    #2b
    f = lambda x:x**9-45*x**8+900*x**7-10500*x**6+78750*x**5-393750*x**4+1312500*x**3-2812500*x**2+3515625*x-1953125
    a = 4.82
    b = 5.2
    tol = 1e-4
    [astar,ier,count] = bisection(f,a,b,tol)
    print('Homework 2b), f(x) = (x-5)^9, expanded form')
    print('the approximate root is',astar)
    if ier != 0:
      print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'the number of iterations used was {count}')
    print('')
    
    #3b
    f = lambda x:x**3+x-4
    a = 1
    b = 4
    tol = 1e-3
    [astar,ier,count] = bisection(f,a,b,tol)
    print('Homework 3b), f(x) = x^3+x-4')
    print('the approximate root is',astar)
    if ier != 0:
      print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'the number of iterations used was {count}')
    print('')
# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier,count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier,count]

def driver_fp():

# test functions 
     f1 = lambda x: -np.sin(2*x)+(5*x/4)-(3/4)
# fixed point is alpha1 = 1.4987....
#fixed point is alpha2 = 3.09... 

     Nmax = 100
     tol = 1e-10

#problem 5
     
     
     x0list = [-1,-0.5,2,3,4.5]
     roots = []
     for x0 in x0list:
          [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
          
          roots.append(xstar)
     #print('the approximate fixed point is:',xstar)
     #print('f1(xstar):',f1(xstar))
     #print('Error message reads:',ier)
     print(f'The roots are {roots}')



# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    
      
driver()        

#prob 5 plot
x = np.linspace(-2.5,10,500)
fx = x-4*np.sin(2*x)-3
plt.plot(x,fx)
plt.hlines(0,-2.5,10,color='red')
plt.title(f'Plot of $f(x)=x-4sin(2x)-3$ and $g(x)=0$')
plt.show()

driver_fp()