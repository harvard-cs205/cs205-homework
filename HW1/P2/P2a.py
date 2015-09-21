from P2 import *

if __name__ == "__main__":
    i_pixels = [i for i in xrange(2000)]
    sc = pyspark.SparkContext()
    pixelrow = sc.parallelize(i_pixels, 10)
    pixels = pixelrow.cartesian(pixelrow)
    pixels = pixels.map(lambda (i, j): ((i, j), (j / 500.0 - 2, i / 500.0 - 2)))
    pixels = pixels.mapValues(lambda (x, y): mandelbrot(x, y))
    # draw_image(pixels)
    #summed_values = sum_values_for_partitions(pixels).collect()
    print pixels.takeSample(True, 20)
    #plt.hist(summed_values, bins=20)
    plt.xlabel("Iterations")
    plt.ylabel("Frequency")
   # plt.show()
