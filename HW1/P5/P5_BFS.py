def bfs_look2(edgelist,from_,to_,dmax,sc):
    global accum 
    dis = 1
    accum = sc.accumulator(1)
    
    def increase_accum(x):
        global accum
        accum += 1
    
    def swap((K,V)):
        V = list(V)
        return (V[0],[K]+V[1:])
        
    nextneigh = sc.parallelize([(from_,0)])
    distance = sc.parallelize([(from_,0)])
    paths = sc.parallelize([(from_,'<-')])


    while dis <= dmax:
        oldaccum = accum.value
        
        temp = edgelist.join(nextneigh).partitionBy(64)
        
        nextneigh = temp.map(lambda (K,V): (V[0],V[1]+1)).distinct().subtractByKey(distance)
        nextneigh.foreach(increase_accum)
        
        temp2 = temp.map(lambda (K,V): (V[0],K)).subtractByKey(distance).map(lambda (K,V): (V,K))
        paths = temp2.join(paths).map(lambda (K,V): swap((K,V)))
        
        if oldaccum == accum.value:
            result = "NO PATH"
            return result
        distance = distance.union(nextneigh).cache()
        result = distance.join(sc.parallelize([(to_,dis)])).partitionBy(64)
        
        if not result.isEmpty():
            result = result.mapValues(lambda v: v[0]).collect()
            result2 = paths.join(sc.parallelize([(to_,dis)])).partitionBy(64).collect()
            break
        dis += 1
        
    return result,result2