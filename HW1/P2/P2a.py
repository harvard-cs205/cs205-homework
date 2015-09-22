# Default partitioning: 100 partitions!

import matplotlib.pyplot as plt

import P2
import seaborn as sns
sns.set_context('poster', font_scale=1.25)
import findspark as fs
fs.init()
import pyspark as ps
sc = ps.SparkContext(appName='P2a')

num_pixels = 2000
rows = sc.range(num_pixels, numSlices=10)
cols = sc.range(num_pixels, numSlices=10)

indices = rows.cartesian(cols)

def mandelbrot_wrapper(row, col):
    x = col/(num_pixels/4.) - 2.
    y = row/(num_pixels/4.) - 2.

    return ((row, col), P2.mandelbrot(x, y))

mandelbrot_rdd = indices.map(lambda a: mandelbrot_wrapper(*a))

# Now collect the data & plot
mandelbrot_result = mandelbrot_rdd.collect()

plt.grid(False)
# I slightly redefined the draw image function as the original
# implementation annoyed me...I did not want to collect in a draw function!
P2.draw_image(data=mandelbrot_result)

plt.savefig('P2a.png', dpi=200, bbox_inches='tight')