# P3_plot.py
# Plot Output
import numpy as np
import matplotlib.pyplot as plt
 
if __name__ == '__main__':
	# Read in data
	results = np.loadtxt('HW3_P3_results2.txt', delimiter=',', dtype=str)

 	# Split data
 	execution_time = [x[-1].split()[0] for x in results]
	workgroups = [x[1].split()[1] for x in results]
	num_workers = [x[2].split()[1] for x in results]

 	### Plot coalesced ###
 	plt.figure(figsize=(10,8))
	
	for x in xrange(6, 43, 6):
		plt.plot(num_workers[x-6:x], execution_time[x-6:x], label='Work Groups: %2s'%(workgroups[x-1]))
		plt.scatter(num_workers[x-6:x], execution_time[x-6:x])
	    
	plt.legend()
	plt.title("Coalesced Sums Execution Time")
	plt.ylabel("Execution Time (seconds)")
	plt.xlabel("Number of Workers")
	plt.ylim(-.01, .18)
	plt.xlim(-2,132)
	plt.show()

	### Plot Block ###
	plt.figure(figsize=(10,8))
	for x in xrange(int(84/2)+6, 85, 6):
	    plt.plot(num_workers[x-6:x], execution_time[x-6:x], label='Work Groups: %2s'%(workgroups[x-1]))
	    plt.scatter(num_workers[x-6:x], execution_time[x-6:x])

	plt.legend()
	plt.title("Blocked Reads Execution Time")
	plt.ylabel("Execution Time (seconds)")
	plt.xlabel("Number of Workers")
	plt.ylim(-.02,.2)
	plt.xlim(-2,132)
	plt.show()