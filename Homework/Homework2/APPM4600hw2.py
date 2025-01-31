# Homework 2 - APPM 4600

# Problem 3c

import numpy as np
import math
import matplotlib.pyplot as plt
import random

def e(x):
    y = math.e**x
    return y - 1

print(e(9.999999995000000 * 1e-10))
print(e(1e-9))

# Problem 4

# Part a)
t = np.linspace(0, np.pi, 31)
y = np.cos(t)
S = np.sum(t * y)

print(f'The sum is: {S}')

# Part b)
# Wavy circle
theta = np.linspace(0, 2*np.pi,100)
R = 1.2
dr = 0.1
f = 15
p = 0

xt = R*(1 + dr*np.sin(f*theta + p))*np.cos(theta)
yt = R*(1 + dr*np.sin(f*theta + p))*np.sin(theta)

plt.plot(xt,yt)
plt.axis("equal")
plt.show()

# for loop

for i in range(10):
    R = i
    dr = 0.05
    f = 2+i
    p = random.uniform(0,2)
    xt = R*(1 + dr*np.sin(f*theta + p))*np.cos(theta)
    yt = R*(1 + dr*np.sin(f*theta + p))*np.sin(theta)
    plt.plot(xt,yt)

plt.axis("equal")
plt.show()