import matplotlib.pyplot as plt
import numpy as np

# Homework 1 APPM 4600

#Problem 1
x = np.arange(1.920,2.080,0.001)
px_coeff = x**9 - 18*x**8 + 144*x**7 - 672*x**6 +2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 +2304*x - 512
px_poly = (x-2)**9

plt.plot(x, px_coeff, label = '$p$ via coefficients')
plt.plot(x,px_poly, label = '$p$ via $(x-2)^9$')
plt.legend()
plt.show()

#Problem 3
error_est = 3*0.5**3/6
print(f'Estimated error upper bound is {error_est}')

actual_error = (1+0.5+0.5**3)*(np.cos(0.5))-(11/8)
print(f'Actual error is {actual_error}')

#Problem 5

x1=np.pi
x2 = 1e6
delta = np.logspace(-16, 0, num=100)
diff1 = np.abs(np.cos(x1+delta) - np.cos(x1) + 2*np.sin((2*x1+delta)/2)* np.sin(delta/2))
diff2 = np.abs(np.cos(x2+delta) - np.cos(x2) + 2*np.sin((2*x2+delta)/2)* np.sin(delta/2))
plt.loglog(delta, diff1, label = 'No Subtraction Error x=$\pi$')
plt.loglog(delta, diff2, label = 'No Subtraction Error x=$10^6$')
plt.legend()
plt.show()


taydiff1 = np.abs(np.cos(x1+delta) - np.cos(x1) + delta*np.sin(x1))
taydiff2  = np.abs(np.cos(x2+delta) - np.cos(x2) + delta*np.sin(x2))
plt.loglog(delta, diff1, label = 'No Subtraction Error x=$\pi$')
plt.loglog(delta, taydiff1, label = 'Taylor Approximation Error x=$\pi$')
plt.legend()
plt.show()

plt.loglog(delta, diff2, label = 'No Subtraction Error x=$10^6$')
plt.loglog(delta, taydiff2, label = 'Taylor Approximation Error x=$10^6$')
plt.legend()
plt.show()

