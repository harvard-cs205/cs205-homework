import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Model parameters
K = 121    # Number of times steps
dt = 0.01  # Delta t
c = 0.1    # Coefficient of drag
g = -9.81  # Gravity
# For constructing matrix B
bx = 0
by = 0
bz = 0
bvx = 0.25
bvy = 0.25
bvz = 0.1
# For constructing matrix C
rx = 1.0
ry = 5.0
rz = 5.0
ax = Axes3D(plt.figure())
s_true = np.loadtxt('P4_trajectory.txt',delimiter=',')
positions = np.zeros([len(s_true),3])
velocities = np.zeros([len(s_true),3])
for itn,row in enumerate(s_true):
    positions[itn] = row[0:3]
    velocities[itn] = row[3:6]
print np.shape(positions)

initial_matrix = np.zeros([3,3])
r_values = [rx,ry,rz]
for itn,row in enumerate(initial_matrix):
    initial_matrix[itn,itn] = 1 / r_values[itn]
r_mat = np.asmatrix(initial_matrix)
print r_mat
print len(positions)

approx_matrix = np.zeros([len(positions),3])
position_mat = np.asmatrix(positions)
print np.shape(approx_matrix)
print approx_matrix
for itn,row in enumerate(positions):
    vertical_mat =  r_mat * position_mat[itn].T
    approx_matrix[itn] = vertical_mat.T
print approx_matrix

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?

    # Initial conditions for s0

A = np.array(np.vstack([np.hstack([1,0,0,dt,0,0]),np.hstack([0,1,0,0,dt,0]),
    np.hstack([0,0,1,0,0,dt]),np.hstack([0,0,0,1-c*dt,0,0]),
    np.hstack([0,0,0,0,1-c*dt,0]),np.hstack([0,0,0,0,0,1-c*dt])]))
print "A = " + str(A)
a = np.array([0,0,0,0,0,g*dt])
print "a = " + str(a)
s0 = np.array([0,0,2,15,3.5,4.0]).T
print "s0 = " + str(s0)
    # Compute the rest of sk using Eq (1)
sk_mat = np.zeros([K,6])
print np.shape(sk_mat)
sk = s0
print np.dot(A,s0)
for itn,row in enumerate(sk_mat):
    sk_mat[itn] = np.dot(A,sk)
    sk = sk_mat[itn]
print sk_mat
simple_x = sk_mat[:,0]
simple_y = sk_mat[:,1]
simple_z = sk_mat[:,2]
ax.plot(simple_x,simple_y,simple_z,
        '-k', label='Blind trajectory')
ax.legend()
plt.show()