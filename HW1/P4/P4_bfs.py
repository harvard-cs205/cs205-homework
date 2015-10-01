def bfs(SOURCE,Graph,sc):
    node1 = [(SOURCE,0)]
    node1 = sc.parallelize(node1).partitionBy(8)
    final_rdd1 = node1
    i = 0
    
    while i < 10:        
        #find neighbors of input node        
        node1 = Graph.join(node1).distinct().values().partitionBy(8) 
        
        #new node only includes neighbors not already searched/in Final RDD        
        node1 = node1.mapValues(lambda x: x+1).subtractByKey(final_rdd1)

        #Add neighbors searched (and their distance value) to the final RDD        
        final_rdd1 = final_rdd1.union(node1).cache()
        i+=1    

    #return count of nodes touched        
    return final_rdd1.groupByKey().mapValues(list).count()

