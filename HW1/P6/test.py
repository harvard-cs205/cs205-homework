#import findspark
#findspark.init()
#print findspark.find()

import pyspark
sc = pyspark.SparkContext(appName="Simple")
#print 'made it'
import numpy as np

tst = [('hi', 'yes')]
print np.random.choice(tst, [1])