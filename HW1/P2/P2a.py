from P2 import *


def compute_image(s, p):

    c = s / 4.0
    ii = sc.parallelize([(i, i/c-2) for i in xrange(s)], p)
    jj = sc.parallelize([(j, j/c-2) for j in xrange(s)], p)

    image = ii.cartesian(jj).map(lambda (k, v):((v[0], k[0]),mandelbrot(k[1], v[1])))
    # Your code here
    draw_image(image)

    partition_work = sum_values_for_partitions(image).collect()
    plt.hist(range(p * p), weights=partition_work, bins=[i-0.5 for i in range(p*p+1)])

    plt.show()

    return image
