
#NOTE: File P5.py should be used to call this function below to find
#shortest path between two nodes


import numpy as np

def connected_components(SOURCE,Graph,sc,):
    node1 = [(SOURCE,0)]
    node1 = sc.parallelize(node1).partitionBy(32)
    final_rdd1 = node1
    components = 1
    #run this while there are always new neighbors to search for
    
    while Graph.subtractByKey(final_rdd1) != 0:
        #find all neighbors and conitnue iterating until no new neighbors found
        while not node1.isEmpty():       
            node1 = Graph.join(node1).distinct().values().partitionBy(32)       
            node1 = node1.mapValues(lambda x: x+1).subtractByKey(final_rdd1)        
            final_rdd1 = final_rdd1.union(node1).partitionBy(32).cache()
        #check to see if characters from graph  not accounted for
        remaining_components = Graph.SubtractByKey(final_rdd1)
        components +=1
        SOURCE = remaining_components.takeSample(False,1)[0]
        node1 = [(SOURCE,0)]
        node1 = sc.parallelize(node1).partitionBy(32)
    return components 

 

  