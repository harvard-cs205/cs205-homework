import findspark
findspark.init()
import pyspark
from pyspark import SparkContext

import time
start = time.time()
def split_link(text):
    colon_split = text.split(": ")
    return (int(colon_split[0]), map(int, colon_split[1].split(" ")))

def iterate_neighbors(node):
    vertex, neighbors = node
    min_neighbor = min(neighbors)
    new_edges = [(min_neighbor, neighbors)]
    for neighbor in neighbors:
        if neighbor != min_neighbor:
            new_edges.append((neighbor, set([min_neighbor])))
    return new_edges

def is_iterated(node):
    vertex, edges = node
    return len(edges) == 1 and vertex not in edges    

def continue_iter(node, accum):
    vertex, edges = node
    if vertex > min(edges):
        accum.add(1)

def reverse_edges(node):
    edges = []
    for neighbor in node[1]:
        edges.append((neighbor, set([node[0]])))
    return edges

def symmetric(rdd):
    rdd = rdd.map(lambda x: (x[0], set([x[0]]) | set(x[1])))
    stop_accum = rdd.context.accumulator(1)
    while stop_accum.value > 0:
        stop_accum = rdd.context.accumulator(0)
        rdd = rdd.flatMap(iterate_neighbors).reduceByKey(lambda x,y: x|y).filter(lambda x: not is_iterated(x))
        rdd.foreach(lambda x: continue_iter(x, stop_accum))
    return {"components": rdd.count(), "max": len(rdd.takeOrdered(1, lambda x: -len(x[1]))[0][1])}

def asymmetric(rdd):
    rdd_reverse = rdd.flatMap(reverse_edges).reduceByKey(lambda x,y: x|y).partitionBy(64)
    rdd_symmetric = rdd.join(rdd_reverse).map(lambda x: (x[0], set(x[1][0]).intersection(x[1][1])))
    return symmetric(rdd_symmetric)

if __name__ == '__main__':
    sc = SparkContext(appName="P5")
    links = sc.textFile('links.txt')
    links_kv = links.map(split_link).partitionBy(64).cache()
    print "Symmetric Links:" + str(symmetric(links_kv))
    print "Asymmetric Links:" + str(asymmetric(links_kv))
 
print time.time() - start   
