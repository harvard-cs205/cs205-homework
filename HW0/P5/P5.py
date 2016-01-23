import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math

#ax = Axes3D(plt.figure())


bags = [i*2 for i in xrange(1, 10)]
time_infinity = [math.log(j)/math.log(2) for j in bags]
time_single = [k-1 for k in bags]

x_coords = bags
y_coords = time_infinity

plt.plot(x_coords, y_coords)
plt.plot(x_coords, time_single)

# ax.plot(x_coords, y_coords,
#         '-r', label='Infinity')

# y_coords = time_single

# ax.plot(x_coords, y_coords,
#         '-b', label='Single')

# ax.legend()

plt.legend(['infinite', 'single'])

plt.show()