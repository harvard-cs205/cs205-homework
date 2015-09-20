def bfs(graph, source, sc):
    dist = 0
    result = sc.parallelize([(source, 0)])
    frontier = result
    resultSize, newResultSize = 1, 0
    closeL = set([source])

    def bfsMapper(x):
        d, adjlist = x[1]
        return [(adj, d+1) for adj in adjlist]

    while resultSize != newResultSize:
        
        frontier = frontier.join(graph) \
                            .flatMap(bfsMapper) \
                            .distinct() \
                            .filter(lambda x: x not in closeL)
        dist += 1
        result = result.union(frontier).reduceByKey(lambda d1, d2: min(d1, d2))
        
        resultSize, newResultSize = newResultSize, 0
        # enforce lazy-evaluation
        for item in result.collect():
            closeL.add(item[0])
            newResultSize += 1

    return result 
  
