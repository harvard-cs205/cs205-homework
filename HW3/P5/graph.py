import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('mathtext', default='regular')

maze_1_total = [213.3, 121.6, 2.8, 8.5]
maze_1_per = [0.233104874317, 0.229895198488, 0.281728, 0.849528]
maze_1_iter = [915, 529, 10, 10]

maze_2_total = [122, 62.7, 2.2, 7.49]
maze_2_per = [0.229861393597, 0.230576764706, 0.244124444444, 0.83256]
maze_2_iter = [531, 272, 9, 9]

x = np.arange(1,5)

fig = plt.figure()
ax = fig.add_subplot(111)

ax2 = ax.twinx()
line1 = ax.plot(x, maze_2_total, '-', label = 'Total Time', color='red')
# line2 = ax2.plot(x, maze_1_iter, '-', label = 'Number of Iterations')
line2 = ax2.plot(x, maze_2_per, '-', label = 'Time per Iteration')

ax.legend(loc=2)
ax2.legend(loc=0)
# lines = line1 + line2
# labs = [l.get_label(l) for l in lines]
# ax.legend(lines, labs, loc=0)

ax.grid()
ax.set_title("Maze 2 Statistics")
ax.set_xlabel("Part")
ax.set_ylabel("Total Time (ms)")
ax2.set_ylabel("Time per Iteration (msmaze")

plt.show()



