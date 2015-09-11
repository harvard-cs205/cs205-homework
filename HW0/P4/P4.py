import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',', usecols=(0,1,2))
    
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    def c_approx(r_value, value):
        return (1/r_value) * float(value)
    
    s_measured = np.loadtxt('P4_measurements.txt', delimiter=',', 
        converters = {0: lambda x: c_approx(rx, x), 
        1: lambda y: c_approx(ry, y), 2: lambda z: c_approx(rz, z)})
    
    ax.plot(s_measured[:,0], s_measured[:,1], s_measured[:,2],
            '.g', label='Observed trajectory')
    
    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    
    A = np.zeros([6, 6])
    A = np.asmatrix(A)
    A[0,0] = 1
    A[1,1] = 1
    A[2,2] = 1
    A[0,3] = dt
    A[1,4] = dt
    A[2,5] = dt
    A[3,3] = 1 - (c * dt)
    A[4,4] = 1 - (c * dt)
    A[5,5] = 1 - (c * dt)
    
    a = np.zeros([6,1])
    a = np.asmatrix(a)
    a[5,0] = g * dt
    
    s0 = np.matrix([0, 0, 2, 15, 3.5, 4.0])
    s0 = np.transpose(s0)

    s = s0
    
    results = np.zeros([6, 0])
    results = np.asmatrix(results)
    results = np.append(results, s0, axis=1)
    
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    for k in range(1, K):
        s = A * s + a
        results = np.append(results, s, axis=1)

    results = np.asarray(results)
    
    ax.plot(results[0,:], results[1,:], results[2,:],
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    
    B = np.zeros([6, 6])
    B = np.asmatrix(B)
    B[0,0] = bx
    B[1,1] = by
    B[2,2] = bz
    B[3,3] = bvx
    B[4,4] = bvy
    B[5,5] = bvz
    
    C = np.zeros([3, 6])
    C = np.asmatrix(C)
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz
    
    sig0 = np.zeros([6, 6])
    sig0 = np.asmatrix(sig0)
    sig0[0,0] = .01
    sig0[1,1] = .01
    sig0[2,2] = .01
    sig0[3,3] = .01
    sig0[4,4] = .01
    sig0[5,5] = .01
    
    s_measured = np.loadtxt('P4_measurements.txt', delimiter=',')
    
    def predictS(sk):
        return A * sk + a
    
    def predictSig(sig):
        return np.linalg.inv(A * sig * A.T + B * B.T)
        
    def updateSig(sig):
        return np.linalg.inv(predictSig(sig) + C.T * C)
    
    def updateS(mk, sig, s):
        return updateSig(sig) * (predictSig(sig) * predictS(s) + C.T * mk)

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    kalman_results = np.zeros([6, 0])
    kalman_results = np.asmatrix(kalman_results)
    kalman_results = np.append(kalman_results, s0, axis=1)
    
    s = s0
    sig = sig0
    s_measured = np.asmatrix(s_measured)
    
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    for k in range(1, K):
        sig = updateSig(sig)
        s = updateS(s_measured[k].T, sig, s)
        kalman_results = np.append(kalman_results, s, axis=1)

    kalman_results = np.asarray(kalman_results)
    
    ax.plot(kalman_results[0,:], kalman_results[1,:], kalman_results[2,:],
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()