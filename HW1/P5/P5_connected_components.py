
#NOTE: File P5.py should be used to call this function below to find
#connected components

import numpy as np

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def connected_components(SOURCE,Graph,sc,):
    node1 = [(SOURCE,0)]
    node1 = sc.parallelize(node1).partitionBy(128)
    final_rdd1 = node1
    Components = []
    Components_sym = []
    Graph2 = Graph.map(lambda (K,V): (V,K))
    Graph_sym = Graph.union(Graph2).distinct().groupByKey().partitionBy(128).cache()
    
    #run this while there are always new neighbors to search for
    
    while Graph.subtractByKey(final_rdd1) != 0:
        #find all neighbors and conitnue iterating until no new neighbors found
        while not node1.isEmpty():       
            
            node1 = Graph.join(node1)
            
            assert copartitioned(node1, Graph)   
            
            node1 = node1.distinct().values().partitionBy(128)
            
            node2 = node1.subtractByKey(final_rdd1)
             
            assert copartitioned(node2, node1)   
            
            node1 = node2
                 
            final_rdd2 = final_rdd1.union(node1)
            assert copartitioned(final_rdd2, final_rdd1)   
            
            final_rdd1=final_rdd2
            final_rdd1.cache()
            
        #check to see if characters from graph  not accounted for
        remaining_components = Graph.SubtractByKey(final_rdd1)
        
        assert copartitioned(remaining_components,Graph) 
        
        Components = Components.append(final_rdd1.groupByKey().count())

        SOURCE = remaining_components.takeSample(False,1)[0]
        
        node1 = [(SOURCE,0)]
        node1 = sc.parallelize(node1).partitionBy(128).cache()



    final_rdd1_sym = node1
    node1_sym = node1
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

    return len(Components), max(Components), len(Components_sym), max(Components_sym)     
    
   

  
