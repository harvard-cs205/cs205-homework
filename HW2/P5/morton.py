import numpy as np

#morton sort
#taken from http://code.activestate.com/recipes/577558-interleave-bits-aka-morton-ize-aka-z-order-curve/
def part1by1(n):
    n&= 0x0000ffff
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def unpart1by1(n):
    n&= 0x55555555
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    n = (n ^ (n >> 4)) & 0x00ff00ff
    n = (n ^ (n >> 8)) & 0x0000ffff
    return n


def interleave2(x, y):
    return part1by1(x) | (part1by1(y) << 1)


def deinterleave2(n):
    return unpart1by1(n), unpart1by1(n >> 1)


def part1by2(n):
    n&= 0x000003ff
    n = (n ^ (n << 16)) & 0xff0000ff
    n = (n ^ (n <<  8)) & 0x0300f00f
    n = (n ^ (n <<  4)) & 0x030c30c3
    n = (n ^ (n <<  2)) & 0x09249249
    return n


def unpart1by2(n):
    n&= 0x09249249
    n = (n ^ (n >>  2)) & 0x030c30c3
    n = (n ^ (n >>  4)) & 0x0300f00f
    n = (n ^ (n >>  8)) & 0xff0000ff
    n = (n ^ (n >> 16)) & 0x000003ff
    return n


def morton_sorter(list_input, precision):
    morton_list = []
    for x in list_input:
            morton_list.append(interleave2(int(x[0]*precision),int(x[1]*precision)))    
    #by the magic of stackoverflow at http://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
    #I have to apply argsort twice. Wow.
    sorter= np.argsort(np.argsort(morton_list))
    return sorter

#Testing code
#test = [(0,0),(0,1),(0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
#test = [(0.1,0.1), (0.1,0.2), (0.1,0.3), (0.2,0.1), (0.2,0.2), (0.2,0.3), (0.3,0.1), (0.3,0.2), (0.3,0.3)]
#sort = morton_sorter(test, 100)
#print sort