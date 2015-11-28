import numpy as np
import math
import pylab as pl

x = xrange(1,100)
y = [math.ceil(math.log(x1,2)) for x1 in x]
z = [x1-1 for x1 in x]
pl.plot(x,y, label="Infinite workers")
pl.plot(x,z, label="Single worker")
pl.xlabel('Bags')
pl.ylabel('Time')
pl.show()
pl.legend(loc='upper center')