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

    s_true = np.loadtxt( "P4_trajectory.txt", delimiter = ',', usecols = ( 0, 1, 2 ) ).transpose()
    ax.plot( s_true[0], s_true[1], s_true[2], '--b', label = 'True trajectory' )

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    # load data
    measurements_matrix = np.mat( np.loadtxt( "P4_measurements.txt", delimiter = ',' ).transpose() )
    dim_stretching_inverse_matrix = np.mat( [ [ 1/rx, 0, 0 ], [ 0, 1/ry, 0 ], [ 0, 0, 1/rz ] ] )
    # compute position estimates
    position_estimate_matrix = dim_stretching_inverse_matrix * measurements_matrix
    
    position_estimate_array = np.array( position_estimate_matrix )
    
    ax.plot( position_estimate_array[0], position_estimate_array[1], position_estimate_array[2], 
             '.g', label='Observed trajectory' )

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initial conditions for s0
    s_initial = np.mat( [ 0, 0, 2, 15, 3.5, 4.0 ] ).transpose()
    A = np.mat( [ [ 1, 0, 0, dt, 0, 0 ],         
                  [ 0, 1, 0, 0, dt, 0 ], 
                  [ 0, 0, 1, 0, 0, dt ], 
                  [ 0, 0, 0, (1 - c*dt), 0, 0 ],
                  [ 0, 0, 0, 0, (1 - c*dt), 0 ],
                  [ 0, 0, 0, 0, 0, (1 - c*dt) ] ] )
    
    a = np.mat( [ [ 0, 0, 0, 0, 0, g*dt ] ] ).transpose()
    
    # Compute the rest of sk using Eq (1)
    
    s_complete = np.mat( np.zeros( shape = ( 6, K ) ) )
    s_complete[:,0] = s_initial
    counter = 1
    
    # total number of measurements is 121 but number of points in trajectory is only 120 (including s0). 
    # I have thus taken 120 steps for a total of 121 position values
    while counter < K:
        s_complete[:,counter] = A * s_complete[:,counter-1] + a
        counter += 1
   
    s_complete = np.array( s_complete )
    
    ax.plot( s_complete[0], s_complete[1], s_complete[2], '-k', label='Blind trajectory' )     

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # Prediction and Updation equations
    
    def PredictS( s_current ) :
        s_intermediate = A * s_current + a
        return s_intermediate
    
    def PredictSig( sigma_current ) :
        sigma_intermediate = np.linalg.inv( A * sigma_current * A.transpose() + B * B.transpose() )
        return sigma_intermediate
    
    def updateSig( sigma_intermediate ) :
        sigma_updated = np.linalg.inv( sigma_intermediate + C.transpose() * C )
        return sigma_updated
    
    def updateS( sigma_updated, sigma_intermediate, s_intermediate, next_measurement ) :
        s_updated = sigma_updated * ( sigma_intermediate * s_intermediate + C.transpose() * next_measurement )
        return s_updated
    
    B = np.mat( [ [ bx, 0, 0, 0, 0, 0 ],         
                  [ 0, by, 0, 0, 0, 0 ], 
                  [ 0, 0, bz, 0, 0, 0 ], 
                  [ 0, 0, 0, bvx, 0, 0 ],
                  [ 0, 0, 0, 0, bvy, 0 ],
                  [ 0, 0, 0, 0, 0, bvz ] ] )

    C = np.mat( [ [ rx, 0, 0, 0, 0, 0 ], 
                  [ 0, ry, 0, 0, 0, 0 ], 
                  [ 0, 0, rz, 0, 0, 0 ] ] )

    # Initial conditions for s0 and Sigma0
    # s0 is unchanged from above --> s0 = s_initial
    sigma_initial = 0.01 * np.mat( np.eye( 6 ) )
            
    # Measurements matrix is unchanged from above --> measurements_matrix
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    
    s_complete = np.mat( np.zeros( shape = ( 6, K ) ) )
    s_complete[:,0] = s_initial
    sigma_current = sigma_initial
    counter = 1
 
    # total number of measurements is 121 but number of points in trajectory is only 120 (including s0). 
    # I have thus taken 120 steps for a total of 121 position values
    while counter < K:
        s_intermediate = PredictS( s_complete[:,counter-1] )
        sigma_intermediate = PredictSig( sigma_current )
        sigma_current = updateSig( sigma_intermediate )
        s_complete[:,counter] = updateS( sigma_current, sigma_intermediate, s_intermediate, measurements_matrix[:,counter] )
        counter += 1
   
    s_complete = np.array( s_complete )
    
    ax.plot( s_complete[0], s_complete[1], s_complete[2], '-r', label='Filtered trajectory' )

    # Show the plot
    ax.legend()
    plt.show()
