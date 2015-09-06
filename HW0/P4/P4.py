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

    s_true = np.loadtxt("P4_trajectory.txt", delimiter=',')
    x_coords = s_true[:,0]
    y_coords = s_true[:,1]
    z_coords = s_true[:,2]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    s_measured = np.transpose(np.loadtxt("P4_measurements.txt", delimiter=','))
    r_matrix = np.array([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]])
    x_tilde = np.transpose(np.dot(r_matrix,s_measured))

    x_coords = x_tilde[:,0]
    y_coords = x_tilde[:,1]
    z_coords = x_tilde[:,2]
    
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?
	
    # Initial conditions for s0
    sk_list = [[0],[0],[2],[15],[3.5],[4.0]] # first column is initial condition
    sk = np.array([[0],[0],[2],[15],[3.5],[4.0]])
    A = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],
            [0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    grav = np.array([[0],[0],[0],[0],[0],[g*dt]])
    
    # Compute the rest of sk using Eq (1)
    for i in range(1,K): # the first column is already done
        # multiply sk by prop and add grav
        sk = np.add(np.dot(A,sk),grav)
        # append the result onto sk_list
        for j,row in enumerate(sk_list):
            row.append(sk[j][0])

    sk_array = np.asarray(sk_list)
    x_coords = sk_array[0]
    y_coords = sk_array[1]
    z_coords = sk_array[2]

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    C = np.array([[rx,0,0,0,0,0],[0,ry,0,0,0,0],[0,0,rz,0,0,0]])
    sk_list = [[0],[0],[2],[15],[3.5],[4.0]]
    sk = np.array([[0],[0],[2],[15],[3.5],[4.0]])
    sig = np.array([[.01,0,0,0,0,0],[0,.01,0,0,0,0],[0,0,.01,0,0,0],
            [0,0,0,.01,0,0],[0,0,0,0,.01,0],[0,0,0,0,0,.01]])
    m_array = s_measured
    mk = m_array[:,0]

    def predictS(sk_prev):
        return np.add(np.dot(A,sk_prev),grav)

    def predictSig(sig_prev):
        B = np.array([[bx,0,0,0,0,0],[0,by,0,0,0,0],[0,0,bz,0,0,0],[0,0,0,bvx,0,0],
            [0,0,0,0,bvy,0],[0,0,0,0,0,bvz]])
        term1 = np.dot(np.dot(A,sig_prev),np.transpose(A))
        term2 = np.dot(B,np.transpose(B))
        return np.linalg.inv(np.add(term1,term2))

    def updateSig(sig_pred):
        return np.linalg.inv(np.add(sig_pred,np.dot(np.transpose(C),C)))

    def updateS(sig_new, sig_pred, sk_pred, mk):
        term1 = np.dot(sig_pred,sk_pred)
        term2 = np.dot(np.transpose(C),mk).reshape(6,1)
        return np.dot(sig_new,np.add(term1,term2))

    for i in range(1,K):
        # compute sk+1
        s_pred = predictS(sk)
        sig_pred = predictSig(sig)
        sig = updateSig(sig_pred)
        sk = updateS(sig, sig_pred, s_pred, mk)
        # append sk+1 onto sk_list
        for j,row in enumerate(sk_list):
            row.append(sk[j][0])
        # update mk
        mk = m_array[:,i]
        
    sk_array = np.asarray(sk_list)
    x_coords = sk_array[0]
    y_coords = sk_array[1]
    z_coords = sk_array[2]
    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
