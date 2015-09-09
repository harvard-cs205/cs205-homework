###############################

# CS 205 Fall 2015 Homework 0 Problem 5
# Submitted by Kendrick Lo (Harvard ID: 70984997)
# Github username: ppgmg

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.arange(1, 500, dtype=np.int)
y = np.log2(x)
z = x - 1

plt.plot(x, y, 'o', label='as many cashiers as you want')
plt.plot(x, z, '.', label='one cashier')
plt.legend(loc = 'lower right')
plt.xlabel("number of bags")
plt.ylabel("counting time (seconds) - log scale")
plt.title("Counting Time for N Bags")

plt.yscale('log')  # show log scale on y-axis

plt.show()

