


#NOTE: File P5.py should be used to call this function below to find
#shortest path between two nodes


import numpy as np

def bfs(SOURCE,Graph,sc,DEST=False):
    node1 = [(SOURCE,0)]
    node1 = sc.parallelize(node1).partitionBy(32)
    final_rdd1 = node1
    
    #while not node1.isEmpty():       
        
        #node1 = Graph.join(node1).distinct().values().partitionBy(16)       
        #node1 = node1.mapValues(lambda x: x+1).subtractByKey(final_rdd1)        
        #final_rdd1 = final_rdd1.union(node1).partitionBy(16).cache()
    
    #final_rdd_g1 = final_rdd1.groupByKey().partitionBy(16).mapValues(list).cache()
    
    #if DEST:
        #return np.min(final_rdd_g1.lookup(DEST)), final_rdd_g1.count()
    #else:
        #return final_rdd_g1.count()

    at_destination = False
    while not at_destination:
        node1 = Graph.join(node1).partitionBy(32)
        node1 = node1.distinct().values().mapValues(lambda x: x+1)
        node1 = node1.subtractByKey(final_rdd1)        
        final_rdd1 = final_rdd1.union(node1).partitionBy(32).cache()
        if final_rdd1.lookup(DEST):
           at_destination = True
    final_rdd_g1 = final_rdd1.groupByKey().mapValues(list).partitionBy(32)
    return np.min(final_rdd_g1.lookup(DEST))
    
