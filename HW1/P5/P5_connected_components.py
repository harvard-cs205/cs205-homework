
#NOTE: File P5.py should be used to call this function below to find
#connected components.

import numpy as np

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def connected_components(SOURCE,Graph,sc,):
    node1 = [(SOURCE,0)]
    node1 = sc.parallelize(node1).partitionBy(128)
    Components_sym = []
    Graph2 = Graph.map(lambda (K,V): (V,K))
    Graph_sym = Graph.union(Graph2).distinct().groupByKey().partitionBy(128).cache()
    final_rdd1_sym = node1
    node1_sym = node1
    
    #run this while there are always new neighbors to search for    
    while Graph_sym.subtractByKey(final_rdd1_sym) != 0:
        #find all neighbors and conitnue iterating until no new neighbors found
        while not node1_sym.isEmpty():       
            
            node1_sym = Graph_sym.join(node1_sym)
            
            assert copartitioned(node1_sym, Graph_sym)   
            
            node1_sym = node1.distinct().values().partitionBy(128)
            
            node2_sym = node1_sym.subtractByKey(final_rdd1_sym)
             
            assert copartitioned(node2_sym, node1_sym)   
            
            node1_sym = node2_sym
                 
            final_rdd2_sym = final_rdd1_sym.union(node1_sym)
            assert copartitioned(final_rdd2_sym, final_rdd1_sym)   
            
            final_rdd1_sym =final_rdd2_sym
            final_rdd1_sym.cache()            
            
        #check to see if characters from graph  not accounted for
        remaining_components_sym = Graph_sym.SubtractByKey(final_rdd1_sym)
        
        assert copartitioned(remaining_components_sym,Graph_sym) 
        
        Components_sym = Components_sym.append(final_rdd1_sym.groupByKey().count())

        SOURCE = remaining_components_sym.takeSample(False,1)[0]
        
        node1_sym = [(SOURCE,0)]
        node1_sym = sc.parallelize(node1_sym).partitionBy(128).cache()

    return len(Components_sym), max(Components_sym)     
    
   

  
