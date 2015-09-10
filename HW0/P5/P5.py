#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt


bagcount = np.linspace(1, 256)


plt.plot(bagcount, bagcount - 1, '--b', label='lone cashier') # using one cashier
plt.plot(bagcount, np.log(bagcount) / np.log(2), '-r', label='infinite many cashiers') # using infinite many cashiers

plt.legend(loc=2)
plt.grid(True)
plt.xlabel('bag count')
plt.ylabel('counting time in seconds')
plt.savefig('P5.png', format='png')
plt.show()

