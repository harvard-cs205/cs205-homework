from pyspark import SparkContext

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

def count_overlap(dist1, dist2, accum):
    accum.add(1)
    return min(dist1, dist2)

def ss_bfs_accum(rdd, root, partitions, diameter = -1):
    visit_rdd = rdd.filter(lambda x: x[0] == root)
    distance_rdd = visit_rdd.map(lambda x: (x[0], 0))
    hops = 1
    count = 1
    while (hops <= diameter or diameter < 0) and count > 0:
        count_accum = rdd.context.accumulator(0)
        visit_rdd = visit_rdd.flatMap(lambda x: x[1]).distinct().cache()
        distance_rdd = visit_rdd.map(lambda x: (x, hops)).union(distance_rdd).reduceByKey(lambda x,y: count_overlap(x,y,count_accum))
        distance_rdd.foreach(lambda x:x)
        count = visit_rdd.count() - count_accum.value
        visit_rdd = rdd.join(visit_rdd.map(lambda x: (x, [])).partitionBy(partitions)).map(lambda x: (x[0], x[1][0]))
        hops += 1
    return distance_rdd.count()


