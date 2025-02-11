################################################################################
# This python script presents examples regarding the newton method and its
# application to 1D nonlinear root-finding, as presented in class.
# APPM 4650 Fall 2021
################################################################################
# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;

# First, we define a function we will test the Newton method with. For each
# function we define, we also define its derivative.
# Our test function from previous sections
def fun(x):
    return x + np.cos(x)-3;
def dfun(x):
    return 1 - np.sin(x);

################################################################################
# We now implement the Newton method
def newton_method(f,df,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0;
    rn=np.array([x0]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            fprintf('\n derivative at initial guess is near 0, try different x0 \n');
    else:
        n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)));

            pn = - fn/dfn; #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn;

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            dfn=df(xn);
            fn=f(xn);
            nfun+=2;

        r=xn;

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)
################################################################################
# Now, we apply this method to our test function
(r,rn,nfun)=newton_method(fun,dfun,3,1e-14,100,True);

# We plot n against log10|f(rn)|
plt.plot(np.arange(0,rn.size),np.log10(np.abs(fun(rn))),'r-o');
plt.xlabel('n'); plt.ylabel('log10|f(rn)|');
plt.suptitle("Newton method results");
plt.show();
input();
################################################################################
# Modes of 'failure' of the Newton method
# We now demonstrate that the Newton method can fail in the following ways:
# (1) Converge linearly due to repeated / multiple roots (Can be fixed)
# (2) Take a while to reach the basin of quadratic convergence
# (3) Fail to converge at all (cycle or diverge)
# Newton methods have to be paired with safeguards to become fully robust

# Double root example (at x=0)
def fun2(x):
    return np.power(x,2)*(np.cos(x)+2);
def dfun2(x):
    return (2*x)*(np.cos(x)+2) + np.power(x,2)*(-np.sin(x));

(r2,rn2,nfun2)=newton_method(fun2,dfun2,0.5,1e-14,100,True);
plt.plot(np.arange(0,rn2.size),np.log10(np.abs(rn2)),'r-o');
plt.xlabel('n'); plt.ylabel('log10|rn|');
plt.suptitle("Newton method results (double root at x=0)");
plt.show();
input();
# Fix for the double root (apply Newton to f(x)/f'(x))
def fun22(x):
    return fun2(x)/dfun2(x);
def d2fun(x):
    return 2*(np.cos(x)+2) + 4*x*(-np.sin(x)) - np.power(x,2)*np.cos(x);
def dfun22(x):
    return 1 - fun2(x)*d2fun(x)/np.power(dfun2(x),2);

(r22,rn22,nfun22)=newton_method(fun22,dfun22,1,1e-14,100,True);
plt.plot(np.arange(0,rn22.size),np.log10(np.abs(rn22)),'r-o');
plt.xlabel('n'); plt.ylabel('log10|rn|');
plt.suptitle("Newton method results (fixed double root at x=0)");
plt.show();
input();

# Example with cyclical behavior (cubic)
def fun3(x):
    return np.power(x,3)-2*x+2;
def dfun3(x):
    return 3*np.power(x,2)-2;

(r3,rn3,nfun3)=newton_method(fun3,dfun3,0.1,1e-14,20,True);
plt.plot(np.arange(0,rn3.size),np.log10(np.abs(fun3(rn3))),'g-^');
plt.xlabel('n'); plt.ylabel('log10|f(rn)|');
plt.suptitle("Newton method results (cyclical behavior)");
plt.show();
input();

# Example where Newton takes its time to get to basin of quadratic convergence
def fun4(x):
    return x + np.cos(2*x) - 3;
def dfun4(x):
    return 1 - 2*np.sin(2*x);

(r4,rn4,nfun4)=newton_method(fun4,dfun4,1,1e-14,200,True);
plt.plot(np.arange(0,rn4.size),np.log10(np.abs(fun4(rn4))),'g-^');
plt.xlabel('n'); plt.ylabel('log10|f(rn)|');
plt.suptitle("Newton method results (slow outside quadratic convergence basin)");
plt.show();
input();

# Example where Newton diverges (derivative has singularity at root)
def fun5(x):
    return np.cbrt(x);
def dfun5(x):
    return (1/3)/np.power(np.cbrt(x),2);

(r5,rn5,nfun5)=newton_method(fun5,dfun5,0.5,1e-14,1000,True);
plt.plot(np.arange(0,rn5.size),np.log10(np.abs(rn5)),'g-^');
plt.xlabel('n'); plt.ylabel('log10|rn|');
plt.suptitle("Newton method results (divergence from root at x=0)");
plt.show();
input();
