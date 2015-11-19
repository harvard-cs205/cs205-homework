import matplotlib.pyplot as plt
import math

# Reference: http://stackoverflow.com/questions/8887544/making-square-axes-plot-with-log2-scales-in-matplotlib

x_coords = []
y_coords = []
single_coords = []

for n in xrange(1, 100000):
    x_coords.append(n)
    y_coords.append( math.ceil( math.log(n,2) ) )
    single_coords.append(n-1)

fig, ax = plt.subplots()

ax.plot(x_coords, y_coords, '-b', label='Infinite employees')
ax.plot(x_coords, single_coords, '-r', label='1 employee')
ax.set_yscale('log', basey=2)
ax.set_xscale('log', basex=2)
plt.xlabel('Number of bags')
plt.ylabel('Total counting time')
plt.title('Counting time')
plt.legend()
plt.show()