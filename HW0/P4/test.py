import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

s_true = np.loadtxt('P4_trajectory.txt',delimiter=',')
positions = np.zeros([len(s_true),3])
velocities = np.zeros([len(s_true),3])
for itn,row in enumerate(s_true):
	positions[itn] = row[0:3]
	velocities[itn] = row[3:6]

print positions, velocities
ax = Axes3D(plt.figure())
ax.plot(positions[:,0],positions[:,1],positions[:,2])
ax.legend()
plt.show()