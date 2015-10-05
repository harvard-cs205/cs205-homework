#NOTE: File P5.py should be used to call this function below to find
#shortest path between two nodes

import numpy as np

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner
    
sequence = []
  
def bfs(SOURCE,Graph,sc,DEST=False):
    node1 = [(SOURCE,0)]
    node1 = sc.parallelize(node1).partitionBy(128)
    final_rdd1 = node1    
    sequence.append(SOURCE) 
    at_destination = False
    while not at_destination:
        
        node1 = Graph.join(node1)
        assert copartitioned(node1, Graph)        
        
        node1 = node1.distinct().values().mapValues(lambda x: x+1).partitionBy(128)
        
        node2 = node1.subtractByKey(final_rdd1)                   
        assert copartitioned(node2,node1)
        
        node1 = node2
        
        final_rdd2 = final_rdd1.union(node1)
        assert copartitioned(final_rdd2,final_rdd1)
        
        final_rdd1 = final_rdd2
        final_rdd1.cache()
        
        if final_rdd1.lookup(DEST):
           at_destination = True
    
    final_rdd_g1 = final_rdd1.groupByKey().mapValues(list)
    final_rdd_g1.cache()
    
    distance = np.min(final_rdd_g1.lookup(DEST))
    if distance == 2:
        RDD1 = final_rdd1.filter(lambda (K,V): V == (distance - 1)).partitionBy(128)
        RDD2 = Graph.join(RDD1)
        RDD3 = RDD2.filter(lambda (K,V): V[0] == DEST)
        RDD4 = RDD3.keys()
        sequence.append(RDD4.collect()[0])
        sequence.append(DEST)
  
    if distance == 3:
        RDD1 = final_rdd1.filter(lambda (K,V): V == (distance - 1)).partitionBy(128)
        RDD2 = Graph.join(RDD1)
        RDD3 = RDD2.filter(lambda (K,V): V[0] == DEST)
        RDD4 = RDD3.keys().collect()[0]
        RDD5 = final_rdd1.filter(lambda (K,V): V == (distance - 2)).partitionBy(128)
        RDD6 = Graph.join(RDD5)
        RDD7 = RDD6.filter(lambda (K,V): V[0] == RDD4)
        RDD8 = RDD7.keys().collect()[0]
        sequence.append(RDD4)
        sequence.append(RDD8)
        sequence.append(DEST)
        
        
    return np.min(final_rdd_g1.lookup(DEST)), sequence
    #return np.min(final_rdd_g1.lookup(DEST))