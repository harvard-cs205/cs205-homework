from math import ceil, log
import numpy as np

def point_to_hilbert(x, y, grid_spacing):
    """ See the extremely helpful description at http://blog.notdot.net/2009/11/Damn-Cool-Algorithms-Spatial-indexing-with-Quadtrees-and-Hilbert-Curves for an explanation of this code. """

    # Tells us how to progress along the hilbert curve
    hilbert_map = {
    'a': {(0, 0): (0, 'd'), (0, 1): (1, 'a'), (1, 0): (3, 'b'), (1, 1): (2, 'a')},
    'b': {(0, 0): (2, 'b'), (0, 1): (1, 'b'), (1, 0): (3, 'a'), (1, 1): (0, 'c')},
    'c': {(0, 0): (2, 'c'), (0, 1): (3, 'd'), (1, 0): (1, 'c'), (1, 1): (0, 'b')},
    'd': {(0, 0): (0, 'a'), (0, 1): (3, 'c'), (1, 0): (1, 'd'), (1, 1): (2, 'd')},
    }

    # Figure out how many points we have in our grid - we are on a [0, 1] x [0, 1] grid
    size_of_grid = 1 / grid_spacing

    # Now get the order of the hilbert curve we need - essentially the closest power of 2 that is >= to the number
    # of grid points that we have
    order = int(ceil(log(size_of_grid) / log(2)))

    # Now compute the hilbert curve
    current_square = 'a'
    position = 0

    for ii in range(order-1, -1, -1):
        position <<= 2
        quad_x = 1 if x & (1 << ii) else 0
        quad_y = 1 if y & (1 << ii) else 0
        quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
        position |= quad_position

    return position
