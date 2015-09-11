import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math as ma

x = np.linspace(1,10)
y = [ma.log(i/2,2)+1 for i in x]
y2 = x-1
plt.plot(x, y, label = 'Infinate Employees')
plt.plot(x, y2, label = 'One Employee')
plt.xlabel('Number of Bags')
plt.ylabel('Wait Time')
plt.title('Speedup versus number of bags with inf employees')
plt.legend(loc = 'best')
plt.savefig('P5.png')
plt.show()