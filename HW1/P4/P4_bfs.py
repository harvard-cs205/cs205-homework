from pyspark import SparkContext
from pyspark import AccumulatorParam

#helper functions for creating the graph from the original source csv
#creating rdd's with keys of comics first mapping to a character and array of that character
#then reduce the one with the array to create a list of characters in each comic
#then join with the single character rdd to form a (comic, character, [characters]) tuple
#then cut off the comic and reduce by character to form character -> [characters]

def comic_KV(x):
    K_V = x.split('","')
    return (K_V[1][0:-1], [K_V[0][1:]])
def node_KV(x):
    K_V = x.split('","')
    return (K_V[1][0:-1], K_V[0][1:])
def get_neighbors(val):
    c_n = val[1]
    neighbors = c_n[1][:]
    neighbors.remove(c_n[0])
    return (c_n[0], neighbors)
def group_neighbors(x, y):
    return set(x) | set(y)

#custom Accumulator for storing hash instead of just a number
#will store superhero -> distance from root
class DistanceAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return initialValue

    def addInPlace(self, v1, v2):
        for key in v2.keys():
            if key not in v1:
                v1[key] = v2[key]
        return v1
    
#bfs search on rdd given root name using the custom Accumulator
#optional parameter of diameter
def ss_bfs_accum(rdd, root, diameter = -1):
    distance_hash = rdd.context.accumulator({root: 0}, DistanceAccumulatorParam())
    next_hop = rdd.lookup(root)[0]
    hops = 1
    while (hops <= diameter or diameter < 0) and len(next_hop) > 0:
        next_rdd = rdd.filter(lambda x: x[0] in next_hop)
        next_rdd.foreach(lambda x: distance_hash.add({x[0] : hops}))
        next_hop = set(next_rdd.flatMap(lambda x: x[1]).collect()) - set(distance_hash.value.keys())
        hops += 1
    return distance_hash.value

#bfs search on rdd given root name without Accumulator, turned out to be faster
def ss_bfs(rdd, root, diameter = -1):
    next_hop = rdd.lookup(root)[0]
    h = {root: 0}
    hops = 1
    while (hops <= diameter or diameter < 0) and len(next_hop) > 0:
        for characters in next_hop:
            if characters not in h:
                h[characters] = hops
        next_hop = set(rdd.filter(lambda x: x[0] in next_hop).flatMap(lambda x: x[1]).collect()) - set(h.keys())
        hops += 1
    return h


