import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, Video
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

def driver():
    ############################################################################
    ############################################################################
    # Rootfinding example start. You are given F(x)=0.

    #First, we define F(x) and its Jacobian.
    def F(x):
        return np.array([(x[0]-4)**2+2*(x[1]-2)**2-32 , x[1]*(x[0]-2)-16 ]);
    def JF(x):
        return np.array([[2*(x[0]-4),4*(x[1]-2)],[x[1],(x[0]-2)]]);

    # Apply Newton Method:
    x0 = np.array([4.0,4.0]); tol=1e-14; nmax=100;
    (rN,rnN,nfN,nJN) = newton_method_nd(F,JF,x0,tol,nmax,True);
    print(rN)

    # Apply Lazy Newton (chord iteration)
    nmax=1000;
    (rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F,JF,x0,tol,nmax,True);

    # Apply Broyden Method
    Bmat='fwd'; B0 = JF(x0); nmax=100;
    (rB,rnB,nfB) = broyden_method_nd(F,B0,x0,tol,nmax,Bmat,True);

    # Plots and comparisons
    numN = rnN.shape[0];
    errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.title('Newton iteration log10|r-rn|');
    plt.legend();
    plt.show();

    numB = rnB.shape[0];
    errB = np.max(np.abs(rnB[0:(numB-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.plot(np.arange(numB-1),np.log10(errB+1e-18),'g-o',label='Broyden');
    plt.title('Newton and Broyden iterations log10|r-rn|');
    plt.legend();
    plt.show();

    numLN = rnLN.shape[0];
    errLN = np.max(np.abs(rnLN[0:(numLN-1)]-rN),1);
    plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
    plt.plot(np.arange(numB-1),np.log10(errB+1e-18),'g-o',label='Broyden');
    plt.plot(np.arange(numLN-1),np.log10(errLN+1e-18),'r-o',label='Lazy Newton');
    plt.title('Newton, Broyden and Lazy Newton iterations log10|r-rn|');
    plt.legend();
    plt.show();

    ############################################################################
    # Same scripts as Newton, but with Broyden
    #Intersection of two circles example:
    def F2(x):
        v = np.array([(x[0]-1)**2 + x[1]**2 -1,
                 (x[0]-2)**2 + (x[1]-1)**2 -1])
        return v;
    def JF2(x):
        M = np.array([[2*(x[0]-1),2*x[1]],[2*(x[0]-2),2*(x[1]-1)]]);
        return M;

    ##########################################################################
    # Convergence basins / number of iterates
    nX=161;nY=161;
    (xx,yy) = np.meshgrid(np.linspace(-0.05,4,nX),np.linspace(-0.95,3,nY));
    xx = xx.flatten(); yy=yy.flatten();
    N = nX*nY;
    ra = np.zeros((N,2));
    nfa = np.zeros(N); nJa=nfa;

    for i in np.arange(N):
        x0 = np.array([xx[i],yy[i]]);
        B0 = JF2(x0); Bmat='fwd';
        if np.linalg.cond(B0)>1e10:
            Bmat='Id';

        (ra[i,:],rn,nfa[i])=broyden_method_nd(F2,B0,x0,1e-14,1000,Bmat,False);

    e1 = np.linalg.norm(ra - np.array([1,1]),axis=1);
    e2 = np.linalg.norm(ra - np.array([2,0]),axis=1);
    plt.rcParams['figure.figsize'] = [6, 5];
    xl=np.linspace(0,4,10); yl=xl-1;
    th=np.linspace(0,2*np.pi,100);
    plt.plot(np.cos(th)+1,np.sin(th));
    plt.plot(np.cos(th)+2,np.sin(th)+1,'r');
    plt.plot(xx[e1<1e-14],yy[e1<1e-14],'g.')
    plt.plot(xx[e2<1e-14],yy[e2<1e-14],'k.')
    plt.plot(xl,yl,'c')
    plt.title("Broyden Convergence Results");
    plt.show();

    ####################################################################
    # 3D iterate plot
    X = xx.reshape(nX,nY);
    Y = yy.reshape(nX,nY);
    Z = 1.0*nfa.reshape(nX,nY);
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    surf=ax.plot_surface(X,Y,np.log10(Z),cmap=cm.coolwarm,linewidth=0, antialiased=False);
    fig.colorbar(surf, shrink=0.5, aspect=5);
    plt.show()

    Z = 1.0*nfa.reshape(nX,nY);
    fig = plt.imshow(np.log10(Z), cmap='hot', interpolation='nearest',extent=[-0.05,4,-0.95,3],origin='lower');
    plt.colorbar(fig);
    plt.show();
    ############################################################################
    # What happens if I always use B0=I?
    nX=161;nY=161;
    (xx,yy) = np.meshgrid(np.linspace(-0.05,4,nX),np.linspace(-0.95,3,nY));
    xx = xx.flatten(); yy=yy.flatten();
    N = nX*nY;
    ra = np.zeros((N,2));
    nfa = np.zeros(N); nJa=nfa;

    for i in np.arange(N):
        x0 = np.array([xx[i],yy[i]]);
        B0 = np.eye(2); Bmat='Id';

        (ra[i,:],rn,nfa[i])=broyden_method_nd(F2,B0,x0,1e-14,1000,Bmat,False);

    e1 = np.linalg.norm(ra - np.array([1,1]),axis=1);
    e2 = np.linalg.norm(ra - np.array([2,0]),axis=1);
    plt.rcParams['figure.figsize'] = [6, 5];
    xl=np.linspace(0,4,10); yl=xl-1;
    th=np.linspace(0,2*np.pi,100);
    plt.plot(np.cos(th)+1,np.sin(th));
    plt.plot(np.cos(th)+2,np.sin(th)+1,'r');
    plt.plot(xx[e1<1e-14],yy[e1<1e-14],'g.')
    plt.plot(xx[e2<1e-14],yy[e2<1e-14],'k.')
    plt.plot(xl,yl,'c')
    plt.show();

    ####################################################################
    # 3D iterate plot
    X = xx.reshape(nX,nY);
    Y = yy.reshape(nX,nY);
    Z = 1.0*nfa.reshape(nX,nY);
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    surf=ax.plot_surface(X,Y,np.log10(Z),cmap=cm.coolwarm,linewidth=0, antialiased=False);
    fig.colorbar(surf, shrink=0.5, aspect=5);
    plt.show();

    Z = 1.0*nfa.reshape(nX,nY);
    fig = plt.imshow(np.log10(Z), cmap='hot', interpolation='nearest',extent=[-0.05,4,-0.95,3],origin='lower');
    plt.colorbar(fig);
    plt.show();
    ###########################################################################
    # Examples as size of the system increases
    MM = np.array([50,100,200,400,800,1600,3200,6400]);
    n=0;
    NewtonT=np.zeros(MM.size);
    BroydenT1=np.zeros(MM.size);
    BroydenT2=np.zeros(MM.size);

    for M in MM:

        TOL=1e-10;
        U = rand.random_sample((M,M));
        (Q,R)=np.linalg.qr(U);
        def FunM(x):
            return x**3+0.1*Q@x+3;
        def JFunM(x):
            return np.diag(3*(x**2),0)+0.1*Q;

        Nexp=10;
        for j in np.arange(Nexp):
            start = timer()
            x0 = 0.5*np.ones(M) + 0.1*(2*np.random.random(M)-1);
            (r,rn,nf,nJ)=newton_method_nd_LS(FunM,JFunM,x0,TOL,200,verb=True);
            end = timer();
            NewtonT[n]=NewtonT[n] + (end - start);
            print(M)

            start = timer()
            (r,rn,nf)=broyden_method_ndLS(FunM,JFunM(x0),x0,TOL,200,'fwd',verb=True);
            end = timer();
            BroydenT1[n]=BroydenT1[n] +(end - start);

            start = timer()
            (r,rn,nf)=broyden_method_ndLS(FunM,np.eye(M),x0,TOL,200,'Id',verb=True);
            end = timer();
            BroydenT2[n]=BroydenT2[n]+(end - start);

        NewtonT[n] = NewtonT[n]/Nexp;
        BroydenT1[n] = BroydenT1[n]/Nexp;
        BroydenT2[n] = BroydenT2[n]/Nexp;
        print(M)
        print(NewtonT[n])
        print(BroydenT1[n])
        print(BroydenT2[n])
        n+=1;

    # We plot log(M) vs log(Time) and estimate m such that Time ~ C*N^m
    plt.plot(np.log10(MM),np.log10(NewtonT),'k-o',label='Newton');
    n0=4; nf=9;
    (m,b)=np.polyfit(np.log10(MM[n0:nf]),np.log10(NewtonT[n0:nf]),1);
    xl=np.linspace(np.log10(MM[0]),np.log10(MM[-1]),10); yl=m*xl+b;
    plt.plot(xl,yl,'r');
    plt.legend();
    print(m);
    plt.show();

    plt.plot(np.log10(MM),np.log10(NewtonT),'k-o',label='Newton');
    plt.plot(np.log10(MM),np.log10(BroydenT1),'b-o',label='Broyden B0=J_F(x0)');
    (m,b)=np.polyfit(np.log10(MM[n0:nf]),np.log10(BroydenT1[n0:nf]),1);
    xl=np.linspace(np.log10(MM[0]),np.log10(MM[-1]),10); yl=m*xl+b;
    plt.plot(xl,yl,'r');
    plt.legend();
    print(m);
    plt.show();

    plt.plot(np.log10(MM),np.log10(NewtonT),'k-o',label='Newton');
    plt.plot(np.log10(MM),np.log10(BroydenT1),'b-o',label='Broyden B0=J_F(x0)');
    plt.plot(np.log10(MM),np.log10(BroydenT2),'g-o',label='Broyden B0=I');
    (m,b)=np.polyfit(np.log10(MM[n0:nf]),np.log10(BroydenT2[n0:nf]),1);
    xl=np.linspace(np.log10(MM[0]),np.log10(MM[-1]),10); yl=m*xl+b;
    plt.plot(xl,yl,'r');
    plt.legend();
    print(m);
    plt.show();

################################################################################
# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Lazy Newton method (chord iteration) in n dimensions implementation
def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn);

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn);

    n=0;
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Implementation of Broyden method. B0 can either be an approx of Jf(x0) (Bmat='fwd'),
# an approx of its inverse (Bmat='inv') or the identity (Bmat='Id')
def broyden_method_nd(f,B0,x0,tol,nmax,Bmat='Id',verb=False):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1;
    npn=1;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.12f|" % (n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        #Broyden step xn = xn -B_n\Fn
        dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        # Update xn
        xn = xn + dn;
        npn=np.linalg.norm(dn);

        ###########################################################
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        Fn1 = f(xn);
        dFn = Fn1-Fn;
        nf+=1;
        I0rn = Inapp(Bapp,Bmat,Un,Vn,dFn); #In^{-1}*(Fn+1 - Fn)
        un = dn - I0rn;                    #un = dn - In^{-1}*dFn
        cn = dn.T @ (I0rn);                # We divide un by dn^T In^{-1}*dFn
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,dn)));
        Un = np.vstack((Un,(1/cn)*un));

        n+=1;
        Fn=Fn1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)

def LS_Gw(f,xn,Fn,dn,nf,eps,maxbis,verb,LS):
    #Derivative-free linesearch for rootfinding
    #Newton and Quasi-Newton methods (Griewank LS method)

    # Begin line search. Evaluate Fn at full step
    Fnp = f(xn+dn);
    nf+=1;
    beta=1;
    ndn = np.linalg.norm(dn);

    if (LS and ndn > 1e-10):
        dFn = Fnp-Fn; #difference in function evals
        nrmd2 = dFn.T @ dFn; #|Fn|^2 = <Fn,Fn>
        q = -(Fn.T @ dFn)/nrmd2; #quality measure q

        #if verb:
        #    print("q0=%1.1e, beta0 = %1.1e" %(q,beta));

        bis=0;
        while q<0.5+eps and bis<maxbis:
            beta=0.5*beta; #halve beta and try again
            Fnp = f(xn+beta*dn);
            dFn = Fnp-Fn;
            nf+=1;
            nrmd2 = dFn.T @ dFn; #|Fn|^2 = <Fn,Fn>
            q = -(Fn.T @ dFn)/nrmd2; #quality measure q
            bis+=1; #increase bisection counter

    pm = beta*dn;
    nrmpn = beta*ndn;
    xn = xn+beta*dn;
    Fn = Fnp;

    return (xn,Fn,nrmpn,nf,beta);

def broyden_method_ndLS(f,B0,x0,tol,nmax,Bmat='Id',verb=False,LS=True):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    nrmpn = 1;
    n=0;
    nf=1;

    #linesearch parameters
    maxbis=6; eps=1e-5;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;
    beta=1; type='broyden';

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|---beta---|--------|---nfv---|");

    while nrmpn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.12f|%1.3f|%s|%d" % (n,np.linalg.norm(xn),np.linalg.norm(Fn),beta,type,nf));

        #Broyden step xn = xn -B_n\Fn
        if (n==0):
            dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        elif (n==1):
            dn = -IFnp - Un.T@(Vn@Fn);
        else:
            dn = -IFnp - (Vn[n-1]@Fn)*Un[n-1];
            #dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);

        ########################################################
        # Derivative-free line search. If full step is accepted (beta=1), this is
        # equivalent to updating xn = xn + dn, Fn = fun(Fn), nrmpn = norm(pn)
        (xn,Fn,nrmpn,nf,beta)=LS_Gw(f,xn,Fn,dn,nf,eps,maxbis,verb,LS);
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        IFnp = Inapp(Bapp,Bmat,Un,Vn,Fn);
        un = (1-beta)*dn + IFnp;
        cn = beta*dn.T @ (dn+IFnp);
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,beta*dn)));
        Un = np.vstack((Un,-(1/cn)*un));

        n+=1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if nrmpn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)

def newton_method_nd_LS(f,Jf,x0,tol,nmax,verb=False,LS=True):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    #linesearch parameters
    maxbis=8; eps=1e-1; beta=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|--beta--|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|%1.3f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn),beta));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);

        ########################################################
        # Derivative-free line search. If full step is accepted (beta=1), this is
        # equivalent to updating xn = xn + dn, Fn = fun(Fn), nrmpn = norm(pn)
        (xn,Fn,npn,nf,beta)=LS_Gw(f,xn,Fn,pn,nf,eps,maxbis,verb,LS);
        ###########################################################

        n+=1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Execute driver
driver()
