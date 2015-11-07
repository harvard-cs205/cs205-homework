def BFS(sc, graph, name):

    # Note: on the Virtual machine, things work best if everything is partitioned into 2.  
    
    # Rdd of name, shortest distance tuples that will be returned.  In each iteration, heros are added to this.  Initialize with the source node, and partition into 2 partitions to match the graph.
    rdd = sc.parallelize([(name, 0)], 2).cache()
    
    # This RDD is used intermediately to build a collection of connections to build on
    namesRdd = rdd

    # Iterator used in the while loop
    i = 0
       
    acc = sc.accumulator(0)    

    while acc.value == 0:

        i+=1

        # Use join to look up the connections of the next round of connections
        namesRdd = graph.join(namesRdd)

        # Use mapValues to take just the list of connections and remove the distance
        namesRdd = namesRdd.mapValues(lambda (x,y): x).partitionBy(2)

        # Take the connections and flatMap them into one big pool to consider
        namesRdd = namesRdd.values().flatMap(lambda x: x)

        # Reduce the connections to distinct entries
        namesRdd = namesRdd.distinct()
        
        # Subtract out the names that are already in the rdd
        namesRdd = namesRdd.subtract(rdd.keys())
        
        if namesRdd.isEmpty(): acc = sc.accumulator(1)   # If nothing was added, change the accumulator value to exit the loop
        else:
            # create tuples of the connection name and the current distance of the iteration
            namesRdd = namesRdd.map(lambda x: (x, i)).partitionBy(2).cache()

            # Union the connection tuples to the existing list, and repartition
            rdd = rdd.union(namesRdd).partitionBy(2).cache()
 
    return rdd
        
