import findspark
findspark.init('/home/shenjeffrey/spark/')
import pyspark
import matplotlib.pyplot as plt
import seaborn as sns
from P2 import *

# initiate spark
sc = pyspark.SparkContext()

# Declare 2000 pixels
pixels = range(2000)

# declare 10 partitions
data = sc.parallelize(pixels, 10)
# cartesian
data = data.cartesian(data)

# Show that the data rdd now has 100 partitions
print data.getNumPartitions()
# Example of the data
data.take(10)

# Partition by uniformly sampling from 0 to 100 to put each Key into 
# one of the 100 partitions
data = data.partitionBy(100, lambda hash_key: np.random.randint(0, 100, size=1)[0])
data.getNumPartitions()

# After rehashing - the ordering of the data is randomly switched!
data.take(10)

# Map the cartesian coordinates as given in the promp
# For the pixel at (i,j), use x = (j/500.0) - 2 and y = (i/500:0) - 2.
data = data.map(lambda (i,j): ((i,j), (j/500.0-2, i/500.0-2)))

# Apply the mandelbot function to each of the cartesian coordinates using mapValue
result = data.mapValues(lambda (x,y): mandelbrot(x,y))

# Draw the image using draw_image
draw_image(result)

# Calculate per-partition work using sum_values_for_partitions function
partition_work = sum_values_for_partitions(result)
# Collect the results
partition_work_result = partition_work.collect()

# Plot histogram
plt.figure(figsize=(10,10))
plt.hist(partition_work_result, bins=15)
plt.ylabel("Counts")
plt.xlabel("Iteration Counts")
plt.show()


