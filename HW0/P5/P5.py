import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
ax = plt.figure()
bag_ct = np.arange(1, 256)
count_time_inf = np.log2(bag_ct)
h_inf = plt.plot(bag_ct, count_time_inf, label="inf employees")
h_one = plt.plot(bag_ct, bag_ct - 1, label="one employee")
plt.xlabel('Bag count (log)')
plt.ylabel('Count time')
plt.xscale('log')
plt.legend(['Inf employees', 'One employee'], loc=2)
plt.savefig('P5.png') 
