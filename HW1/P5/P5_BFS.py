#BFS algorithm with find path feature:
# Function takes 5 argument:
# - The graph in the form of an edgelist
# - The starting link (integer parsed as string)
# - The target link (integer parsed as string)
# - The maximum number of iteration in case there is an issue, should normally stop when no more nodes can be touched
# - the spark context

#Output of the function is a quite intricate tuple (sorry couldn't figure a way to make it more readable):
# - K: (target,distance)
# - V: [(node_to,([intermediate_node_n,…[node_from,<-]],distance)…)]
#



def bfs_look(edgelist,from_,to_,dmax,sc):
    global accum 
    dis = 1
    accum = sc.accumulator(1)
    
    def increase_accum(x):
        global accum
        accum += 1
    
    #Function used to format the paths rdd
    def swap((K,V)):
        V = list(V)
        return (V[0],[K]+V[1:])
     
    nextneigh = sc.parallelize([(from_,0)])
    distance = sc.parallelize([(from_,0)])
    paths = sc.parallelize([(from_,'<-')])


    while dis <= dmax:
        print dis
        oldaccum = accum.value
        
        temp = edgelist.join(nextneigh).partitionBy(256)
        
        nextneigh = temp.map(lambda (K,V): (V[0],V[1]+1)).distinct().subtractByKey(distance)
        nextneigh.foreach(increase_accum)
        
        #Up to here same as in the regular BFS in P4

        temp2 = temp.map(lambda (K,V): (V[0],K)).subtractByKey(distance).map(lambda (K,V): (V,K))
        # Find the previous edges going to the members of nextneigh and verify that nodes haven't already been hit
        
        paths = temp2.join(paths).map(lambda (K,V): swap((K,V)))
        # Update the paths rdd by adding the new valid steps and formating it
     
        
        if oldaccum == accum.value:
            result = "NO PATH"
            return result
        distance = distance.union(nextneigh).partitionBy(256).cache()
        
        #Result checks if the target node has been hit in order to stop the loop earlier:
        result = distance.join(sc.parallelize([(to_,dis)])).partitionBy(256)

        if not result.isEmpty():
            result = result.mapValues(lambda v: v[0]).collect()
            result2 = paths.join(sc.parallelize([(to_,dis)])).partitionBy(256).collect()
            break
        dis += 1
        
    return result,result2