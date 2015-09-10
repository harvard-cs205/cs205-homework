import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv


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
    # Read the file
    trueFile = open("P4_trajectory.txt","r"); 
    x_coords, y_coords, z_coords = np.loadtxt(trueFile,delimiter =',',usecols = (0,1,2),unpack = True);
    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    # Read the file
    ObsFile = open("P4_measurements.txt","r");
    x,y,z = np.loadtxt(ObsFile, delimiter = ',', unpack = True);
    # Constract Measurement array 
    M = np.array([x,y,z]);
    x_coords = (1/rx) * M[0];
    y_coords = (1/ry) * M[1];
    z_coords = (1/rz) * M[2];    
    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,(1-c*dt),0,0],[0,0,0,0,(1-c*dt),0],[0,0,0,0,0,(1-c*dt)]]);
    a = np.matrix([[0,0,0,0,0,(g*dt)]]);
    a = a.transpose();
    s0 = np.matrix([[0,0,2,15,3.5,4.0]])
    s0 = s0.transpose();
    #Construct a 6 * K matrix
    S = np.empty(shape=(6,K), dtype = float)
    # Fill in all the columns 
    # Initial conditions for s0
    S[:,0] = s0.transpose();
    # Set previous array equals to 0
    s_p = s0;
    # Compute the rest of sk using Eq (1)
    for x in range(1,K):
        s_k = A * s_p + a;
        S[:,x] = s_k.transpose();
        s_p = s_k;

    # Filled all 6*K matrix
    # Plot the first three row in the array
    x_coords = S[0,:];
    y_coords = S[1,:];
    z_coords = S[2,:];
     
    # Print x_coords, y_coords, z_coords
    ax.plot(x_coords, y_coords, z_coords,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    # Create the B Matrix
    B = np.identity(6);
    b = np.matrix([[bx,by,bz,bvx,bvy,bvz]]);
    B = b * B;

    # Create the C Matrix ( 3 * 6)
    C = np.zeros(shape = (3,6));
    C[0,0] = rx;
    C[1,1] = ry;
    C[2,2] = rz;
    C = np.asmatrix(C);# Convert to matrix

    # Initial conditions for s0 and Sigma0
    # find Sigma0
    sigma0 = 0.01 * np.identity(6);
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    # Compute the new array X
    X = np.empty(shape=(6,K), dtype = float);
    X[:,0] = s0.transpose();
    s_p = s0;
    sigma_p = sigma0;
    
    # Convert measurement matrix (MEA)
    MEA = np.asmatrix(M);

    for x in range(1,K):
        s_predicted = A * s_p + a;
        sigma_predicted = (A * sigma_p * A.transpose() + B * B.transpose())
        sigma_predicted = inv(sigma_predicted);#inverse the matrix
        sigma_next = sigma_predicted + C.transpose() * C;
        sigma_next = inv(sigma_next);
        s_next = sigma_next * (sigma_predicted * s_predicted + C.transpose()* MEA[:,x]);
        X[:,x] = s_next.transpose();
        s_p = s_next;
        sigma_p = sigma_next;
        print x

    x_coords = X[0,:];
    y_coords = X[1,:];
    z_coords = X[2,:];

    ax.plot(x_coords, y_coords, z_coords,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
