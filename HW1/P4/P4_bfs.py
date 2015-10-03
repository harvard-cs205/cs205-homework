def bfs(graph, source):
    sc = graph.context
    result = sc.parallelize([(source, 0)])
    frontier = result
    resultSize, newResultSize = 1, 0

    def bfsMapper(x):
        d, adjlist = x[1]
        return [(adj, d+1) for adj in adjlist]

    while resultSize != newResultSize:
        resultSize = newResultSize
        frontier = frontier.join(graph) \
                            .flatMap(bfsMapper) \
                            .distinct() \
                            .repartition(min(max(resultSize / 10, 1), 64))
        result = result.union(frontier).reduceByKey(lambda d1, d2: min(d1, d2))
        # enforce lazy-evaluation
        newResultSize = result.count()

    return result 
  
