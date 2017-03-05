import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin

	


def kalman_filter(s_0, Sig_0, A, B, C, K, s_obs):
	# initializations
	Sig_k = Sig_0
	Sig_t = Sig_k
	s_tilde = s_0
	s_f_pred = np.zeros((K, 6))
	s_f_pred[0] = s_0
	
	for i in range(K-1):
		# prediction
		s_tilde = predictS(A, np.asmatrix(s_f_pred[i]).T, a)
		Sig_t = predictSig(A, B, Sig_k)

		#update
		Sig_k = updateSig(Sig_t, C)
		s_f_pred[i+1] = np.asarray(updateS(Sig_k, Sig_t, s_tilde, C, np.asmatrix(s_obs[:,i+1])))
	
	return s_f_pred	

	
def predictS(A, s_k, a):
	s_tilde = A*s_k + a
	return s_tilde
	
def predictSig(A, B, Sig_k):
	Sig_t = (A*Sig_k*A.T + B*B.T).I
	return Sig_t
	
def updateSig(Sig_t, C):
	Sig_succ = (Sig_t + C.T*C).I
	return Sig_succ
	
def updateS(Sig_succ, Sig_t, s_tilde, C, m_succ):
	s_succ = Sig_succ*(Sig_t*s_tilde + C.T*m_succ)
	return s_succ.T
	

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
	rx = 1
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

	# s_true - 120 x 6 array of actual states
	s_true = np.loadtxt('P4_trajectory.txt', dtype=np.float, delimiter=',')	
	# s_obs - 121 x 3 array of observed states (position only)
	s_obs = np.loadtxt('P4_measurements.txt', dtype=np.float, delimiter=',')
	
	x_coords = s_true[:,0] # x-coords is the 0th column
	y_coords = s_true[:,1] # y-coords is the 1st column
	z_coords = s_true[:,2] # z-coords is the 2nd column

	ax.plot(x_coords, y_coords, z_coords, '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

	# convert true and observed states to matrices
	s_true = np.asmatrix(s_true)
	s_obs = np.asmatrix(s_obs)
	
	# define scaling matrix, M, s.t. observed coords is 
	# approx equal to scaled true coords
	scale_mat =  np.diag(np.array([rx, ry, rz]))
	
	# build measurement matrix, C, from scaling matrix. C picks 
	# out coords from state vector and scales them
	C = np.asmatrix(np.hstack((scale_mat, np.zeros((3,3)))))
	
	# computes the inverse of scaling matrix
	scale_mat_I = np.asmatrix(scale_mat).I
	
	# applies inverse scaling to observed coords
	scale_obs = scale_mat_I*(s_obs.T)
	
	x_coords = np.asarray(scale_obs)[0, :]
	y_coords = np.asarray(scale_obs)[1, :]
	z_coords = np.asarray(scale_obs)[2, :]

	
	ax.plot(x_coords, y_coords, z_coords,'.g', label='Observed trajectory')
	

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
	I_3 = np.identity(3)
	UR = dt * I_3
	LL = np.zeros((3, 3))
	LR = I_3 - c*dt*I_3
	
	# build propagation matrix A as block matrix from the above 
	# four blocks
	A = np.bmat('I_3, UR; LL, LR')
	
	# build the downward grav acc vector
	a = np.append(np.zeros((5, 1)), [[g*dt]], axis=0)

	# define initial state vector
	s_0 = np.array([s_true[0]])

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

	# build a K x 6 matrix of predicted position vectors, 
	# initialized with zeros
	s_pred = np.zeros((K, 6))
	
	s_pred[0] = s_0
	
	for i in range(K-1):
		# s_i+1 = A*s_i + a (then convert to row vector)
		s_pred[i+1] = (np.dot(A, s_pred[i].T.reshape(6,1)) + a).T
		
	x_coords = s_pred[:, 0]
	y_coords = s_pred[:, 1]
	z_coords = s_pred[:, 2]

	ax.plot(x_coords, y_coords, z_coords, '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

	# define propagation error scaling matrix
	B = np.diag(np.array([bx, by, bz, bvx, bvy, bvz]))

    # C is as computed from part 2

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

	# initialize Sigma
	Sig_0 = np.asmatrix(0.01*np.identity(6))
	
	# apply kalman filtering to pos estimates
	s_f_pred = kalman_filter(s_0, Sig_0, A, B, C, K, s_obs.T)
	
	x_coords = s_f_pred[:, 0]
	y_coords = s_f_pred[:, 1]
	z_coords = s_f_pred[:, 2]

	ax.plot(x_coords, y_coords, z_coords, '-r', label='Filtered trajectory')

    # Show the plot
	ax.legend()
	plt.show()
