#Single-Source Breadth-First Search:
# Takes 4 arguments:
# - The graph in the form of an edgelist
# - The starting superhero
# - The maximum number of iteration in case there is an issue, should normally stop when no more nodes can be touched
# - the spark context

#It returns both the number of nodes touched INCLUDING the starting node
# and an rdd (not collected) that saves the in the form of a list of tuples (Superhero touched, how far he is)

def bfs(edgelist,SH,dmax,sc):
    global accum 
    accum = sc.accumulator(1)
    
    def increase_accum(x):
        global accum
        accum += 1
        
    dis = 1
    
    nextneigh = sc.parallelize([(SH,0)])
    distance = sc.parallelize([(SH,0)])
    
    while dis <= dmax:
        oldaccum = accum.value
        
        #Finding the next neighbours and excluding the ones that have been already visited:
        nextneigh = edgelist.join(nextneigh).partitionBy(8).map(lambda (K,V): (V[0],V[1]+1)).distinct().subtractByKey(distance)
        
        #Increment the accumulator by counting the number of new neighbours found:
        nextneigh.foreach(increase_accum)
        
        #Break outside the loop, if we didn't find any new neighbours:
        if oldaccum == accum.value:
            break
        
        #add the new neighbours to the result rdd:
        distance = distance.union(nextneigh).partitionBy(8).cache()
        dis += 1
        
    return oldaccum,distance