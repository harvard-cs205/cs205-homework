# import findspark
# findspark.init()
import pyspark
import numpy as np


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def filterer(i):
    def filt(x):
        return x==itr_acc.value
    return filt

def get_first_non_empty_cogroup(x2):
    for elem in x2:
        elem = list(elem)
        if len(elem) > 0:
            return elem[0]

def BFS(graph, starting_node, N):
    
        #We set the starting node to have distance = 0 
    nodes = sc.parallelize([(starting_node, 1)]).partitionBy(N)
    
    graph.partitionBy(N).cache()
    
    total_char_acc = sc.accumulator(1)
    it_acc = sc.accumulator(0)

    while True:
        it_acc += 1
        dist = it_acc.value

        nodes_2 = nodes.filter(lambda x: filterer(x[1])).join(graph).flatMap(lambda x: x[1][1]).distinct().map(lambda x: (x, dist + 1)).partitionBy(N)

        nodes = nodes.cogroup(nodes_2).map(lambda x: (x[0], get_first_non_empty_cogroup(list(x[1])))).partitionBy(N).cache()
        
        assert copartitioned(nodes_2, nodes)

        if nodes.count() == total_char_acc.value:
            break

        total_char_acc += nodes.count() - total_char_acc.value
        
    return nodes.count(), nodes