from P2 import *


def compute_image(s, p):
    """Compute Mandelbrot image using default partitioning strategy.
    :param s: The size of the square image to compute.
    :param p: The square root of the number of partitions.
    """
    c = s / 4.0
    ii = sc.parallelize([(i, i / c - 2) for i in xrange(s)], p)
    jj = sc.parallelize([(j, j / c - 2) for j in xrange(s)], p)

    image = ii.cartesian(jj).map(
        lambda (k, v): ((v[0], k[0]), mandelbrot(k[1], v[1])))

    draw_image(image)

    # create nice evenly spaced bins
    b = [i - 0.5 for i in range(p * p + 1)]
    b[0] = 0
    b[p * p] = p * p

    partition_work = sum_values_for_partitions(image).collect()
    plt.hist(range(p * p), weights=partition_work, bins=b)

    plt.show()

    return image
