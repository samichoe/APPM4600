import scipy as spy
import numpy as np
import matplotlib.pyplot as plt
import math

t = 60*60*24*60
alpha = 0.138*1e-6
print(np.e)
def fun(x,t):
    return 35*spy.special.erf(x/2*np.sqrt(alpha*t)) - 15

denom = 2*np.sqrt(t*alpha)

#print(spy.special.erfinv(3/7)*denom)


def fx(x,t=60*60*24*60):
    return spy.special.erf(x/(2*np.sqrt(alpha*t)))-(3/7)
x = np.linspace(0,0.7,100)
plt.plot(x,fx(x))
plt.axhline(0)
plt.show()


def bisect_method(f,a,b,tol,nmax,vrb=False):
    #Bisection method applied to f between a and b

    # Initial values for interval [an,bn], midpoint xn
    an = a; bn=b; n=0;
    xn = (an+bn)/2;
    # Current guess is stored at rn[n]
    rn=np.array([xn]);
    r=xn;
    ier=0;

    if vrb:
        print("\n Bisection method with nmax=%d and tol=%1.1e\n" % (nmax, tol));

    # The code cannot work if f(a) and f(b) have the same sign.
    # In this case, the code displays an error message, outputs empty answers and exits.
    if f(a)*f(b)>=0:
        print("\n Interval is inadequate, f(a)*f(b)>=0. Try again \n")
        print("f(a)*f(b) = %1.1f \n" % f(a)*f(b));
        r = None;
        return r
    else:
        #If f(a)f(b), we proceed with the method.
        if vrb:
            print("\n|--n--|--an--|--bn--|----xn----|-|bn-an|--|---|f(xn)|---|");
            

        while n<=nmax:
            if vrb:
                print("|--%d--|%1.4f|%1.4f|%1.8f|%1.8f|%1.8f|" % (n,an,bn,xn,bn-an,np.abs(f(xn))));

                ################################################################
                # Plot results of bisection on subplot 1 of 2 (horizontal).
                xint = np.array([an,bn]);
                yint=f(xint);
                ################################################################

            # Bisection method step: test subintervals [an,xn] and [xn,bn]
            # If the estimate for the error (root-xn) is less than tol, exit
            if (bn-an)<2*tol: # better test than np.abs(f(xn))<tol
                ier=1;
                break;

            # If f(an)*f(xn)<0, pick left interval, update bn
            if f(an)*f(xn)<0:
                bn=xn;
            else:
                #else, pick right interval, update an
                an=xn;

            # update midpoint xn, increase n.
            n += 1;
            xn = (an+bn)/2;
            rn = np.append(rn,xn);

    # Set root estimate to xn.
    r=xn;

    if vrb:
        ########################################################################
        # subplot 2: approximate error log-log plot
        e = np.abs(r-rn[0:n]);
        #length of interval
        #ln = (b-a)*np.exp2(-np.arange(0,e.size));
        #log-log plot error vs interval length
        #ax2.plot(-np.log2(ln),np.log2(e),'r--');
        #ax2.set(xlabel='-log2(bn-an)',ylabel='log2(error)');
        ########################################################################

    return r, rn;

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

def newton_2c(f,df,x0,m, tol,nmax,verb=False):

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

            pn = - m*fn/dfn; #modified Newton step
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
            print("Modified newton fixed pt method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Modified newton fixed pt method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)


def bisect_newt(f, df, a, b, btol,ntol, nmax):
    an = a; bn=b; n=0;
    xn = (an+bn)/2;
    # Current guess is stored at rn[n]
    rn=np.array([xn]);
    r=xn;
    ier=0;
    count = 0
    
    # The code cannot work if f(a) and f(b) have the same sign.
    # In this case, the code displays an error message, outputs empty answers and exits.
    if f(a)*f(b)>=0:
        print("\n Interval is inadequate, f(a)*f(b)>=0. Try again \n")
        print("f(a)*f(b) = %1.1f \n" % f(a)*f(b));
        r = None;
        return r
    else:
        while n<=nmax:
            count+=1
            print("|--%d--|%1.4f|%1.4f|%1.8f|%1.8f|%1.8f|" % (n,an,bn,xn,bn-an,np.abs(f(xn))));

            # Bisection method step: test subintervals [an,xn] and [xn,bn]
            # If the estimate for the error (root-xn) is less than tol, exit
            if (bn-an)<2*btol: # better test than np.abs(f(xn))<tol
                ier=1;
                break;

            # If f(an)*f(xn)<0, pick left interval, update bn
            if f(an)*f(xn)<0:
                bn=xn;
            else:
                #else, pick right interval, update an
                an=xn;

            # update midpoint xn, increase n.
            n += 1;
            xn = (an+bn)/2;
            rn = np.append(rn,xn);

    # Set root estimate to xn.
    r=xn;
    print(f'r: {r}')
    rn=np.array([r]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        print('error with derivative')
        return
    else:
        n=0;
        
        print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            
            print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)));

            pn = - fn/dfn; #Newton step
            if np.abs(pn)<ntol or np.abs(fn)<2e-15:
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
            print("Hybrid method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Hybrid method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)

tol = 1e-13
def fun(x):
    return spy.special.erf(x/(2*np.sqrt(0.138*1e-6*60*60*24*60))) - (3/7)


def dfun(x):
    return 2/np.sqrt(np.pi) *np.exp((-x**2)/(4*0.138*1e-6*60*60*24*60))



(r,rn) = bisect_method(fun,0,0.7,tol,100,True)
(r_newt,rn_newt,nfun_newt) = newton_method(fun,dfun,0.01,tol,100,True)
(r_newt,rn_newt,nfun_newt) = newton_method(fun,dfun,0.7,tol,100,True)
print(rn_newt)
x = np.linspace(3,5,100)
def fun2(x):
    return np.e**(3*x) - 27*x**6 + 27*x**4*np.e**x - 9*x**2*np.e**(2*x)

def dfun2(x):
    return 3*np.e**(3*x) - 162*x**5 + 27*x**3*np.e**x*(4+x) - 18*x*np.e**(2*x)*(1+x)

(r_newt,rn_newt,nfun_newt) = newton_method(fun2,dfun2,4,tol,100,True)
(r_newt,rn_newt,nfun_newt) = newton_2c(fun2,dfun2,4,2,tol,100,True)
(r_nb, rnb, nfunb) = bisect_newt(fun2,dfun2,3,5,10e-2,1e-14,100)

def secant_method(f,x0,x1,tol,nmax,verb=False):
    #secant (quasi-newton) method to find root of f starting with guesses x0 and x1

    #Initialize iterates and iterate list
    xnm=x0; xn=x1;
    rn=np.array([x1]);
    # function evaluations
    fn=f(xn); fnm=f(xnm);
    msec = (fn-fnm)/(xn-xnm);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if np.abs(msec)<dtol:
        #If slope of secant is too small, secant will fail. Error message is
        #displayed and code terminates.
        if verb:
            fprintf('\n slope of secant at initial guess is near 0, try different x0,x1 \n');
    else:
        n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|msec|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(msec)));

            pn = - fn/msec; #Secant step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step, update xn-1
            xnm = xn; #xn-1 is now xn
            xn = xn + pn; #xn is now xn+pn

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            fnm = fn; #Note we can re-use this function evaluation
            fn=f(xn); #So, only one extra evaluation is needed per iteration
            msec = (fn-fnm)/(xn-xnm); # New slope of secant line
            nfun+=1;

        r=xn;

        if n>=nmax:
            print("Secant method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Secant method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)




def fun3(x):
    return x**6-x-1

def dfun3(x):
    return 6*x**5-1

(r_newt,rn_newt,nfun_newt) = newton_method(fun3,dfun3,2,tol,100,True)
error_newt = abs(rn_newt-r_newt)
#print(error_newt)

(rs,rns,nfuns)=secant_method(fun3,2,1,tol,100,True)
error_sec = abs(rns-rs)
#print(error_sec)

print("\n|newton error|secant error|")
for i in range(8):
    print("| %1.8f | %1.8f |" %(error_newt[i],error_sec[i]))

plt.loglog(error_newt[1:],error_newt[:-1])
epsilon = 1e-16

# Log-transform the error values
log_error_newt = np.log10(error_newt[1:] + epsilon)
log_error_newt_prev = np.log10(error_newt[:-1] + epsilon)
slope, intercept = np.polyfit(log_error_newt_prev, log_error_newt, 1)
print(f"Slope of the log-log plot for Newton's method: {slope}")
plt.suptitle("Log scale error of Newton's method")
plt.title(f"Slope: {slope}")
plt.show()
#slope, intercept = np.polyfit(np.log(error_newt[1:]), np.log(error_newt[:-1]), 1)
#print(slope)

plt.loglog(error_sec[1:],error_sec[:-1])
log_error_sec = np.log10(error_sec[1:] + epsilon)
log_error_sec_prev = np.log10(error_sec[:-1] + epsilon)
slope, intercept = np.polyfit(log_error_sec_prev, log_error_sec, 1)
print(f"Slope of the log-log plot for Secant method: {slope}")
plt.suptitle("Log scale error of Secant method")
plt.title(f"Slope: {slope}")
plt.show()