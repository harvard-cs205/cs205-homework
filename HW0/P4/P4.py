import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# All of these functions require numpy arrays as input
def predictS(A, s_k, a):
  return np.dot(A,s_k) + a

def predictSig(A, sigk, B):
  return np.linalg.inv(np.dot(np.dot(A, sigk), np.transpose(A)) + np.dot(B, np.transpose(B)))

def updateSig(sig_tild, C):
  return np.linalg.inv(sig_tild + np.dot(np.transpose(C), C))

def updateS(sigk1, sig_tild, s_approx, C, m):
  return np.dot(sigk1, np.dot(sig_tild, s_approx) + np.dot(np.transpose(C), m))


if __name__ == '__main__':
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

    # Create 3D axes for plotting
    ax = Axes3D(plt.figure())

    #####################
    # Part 1:
    #
    # Load true trajectory and plot it
    # Normally, this data wouldn't be available in the real world
    #####################

    traj = np.loadtxt('P4_trajectory.txt', delimiter=',')
    ax.plot(traj[:,0], traj[:,1], traj[:,2],
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    observations = np.loadtxt('P4_measurements.txt', delimiter=',')
    ax.plot((1.0/rx)*observations[:,0], (1.0/ry)*observations[:,1], (1.0/rz)*observations[:,2],
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    # have dt (time step), g (acc. due to grav.), c(drag), K (num time steps)

    s_0 = [0., 0., 2., 15., 3.5, 4.0]
    A = np.array([
        [1.,0.,0.,dt,0.,0.],
        [0.,1.,0.,0.,dt,0.],
        [0.,0.,1.,0.,0.,dt],
        [0.,0.,0.,1.0-c*dt,0.,0.],
        [0.,0.,0,0.,1.0-c*dt,0],
        [0.,0.,0.,0.,0.,1.0-c*dt]
    ])
    a = [0.,0.,0.,0.,0.,g*dt]
    big_mat = []
    for q in range(0,6):
      big_mat.append([0.0]*K)
    big_mat = np.array(big_mat)

    for i in range(6):
      big_mat[i][0] = s_0[i]

    for i in range(1, K):
      res = np.dot(A,big_mat[:,i-1])
      res = [res[k] + a[k] for k in range(6)]
      for j in range(6):
        big_mat[j][i] = res[j]

    ax.plot(big_mat[0], big_mat[1], big_mat[2],
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?

    B = np.array([
      [bx,0.,0.,0.,0.,0.],
      [0.,by,0.,0.,0.,0.],
      [0.,0.,bz,0.,0.,0.],
      [0.,0.,0.,bvx,0.,0.],
      [0.,0.,0.,0.,bvy,0.],
      [0.,0.,0.,0.,0.,bvz]
    ])

    C = np.array([
      [rx,0.,0.,0.,0.,0.],
      [0.,ry,0.,0.,0.,0.],
      [0.,0.,rz,0.,0.,0.]
    ])

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    big_mat_k = []
    for q in range(0,6):
      big_mat_k.append([0.0]*K)
    big_mat_k = np.array(big_mat_k)

    for i in range(6):
      big_mat_k[i][0] = s_0[i]

    sigma_ks = []
    sig0 = 0.01*np.identity(6)
    sigma_ks.append(sig0)

    obs_array = np.array(observations)

    for i in range(1, K):
      tmp_s = predictS(A, big_mat_k[:,i-1], a)
      tmp_sig = predictSig(A, sigma_ks[-1], B)
      new_sig = updateSig(tmp_sig, C)
      sigma_ks.append(new_sig)

      res = updateS(new_sig, tmp_sig, tmp_s, C, obs_array[i])
      for j in range(6):
        big_mat_k[j][i] = res[j]


    ax.plot(big_mat_k[0], big_mat_k[1], big_mat_k[2],
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
