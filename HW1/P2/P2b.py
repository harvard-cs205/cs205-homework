from P2 import *


def compute_image_balanced(s, p):
    """Compute Mandelbrot image using balanced partitioning strategy.
    :param s: The size of the square image to compute.
    :param p: The square root of the number of partitions.
    """
    c = s / 4.0

    ij = sc.parallelize([(ind / s, ind % s) for ind in sorted(
        xrange(s * s), key=lambda k: random.random())], p * p)

    image = ij.map(
        lambda (i, j): ((i, j), mandelbrot(j / c - 2, i / c - 2)))

    draw_image(image)

    # create nice evenly spaced bins
    b = [i - 0.5 for i in range(p * p + 1)]
    b[0] = 0
    b[p * p] = p * p

    partition_work = sum_values_for_partitions(image).collect()
    plt.hist(range(p * p), weights=partition_work, bins=b)

    plt.show()

    return image
