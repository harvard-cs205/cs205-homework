def bfs(graph, source):
    sc = graph.context
    # Act partially as a closed list.
    result = sc.parallelize([(source, 0)])
    # frontier queue
    frontier = result
    resultSize, newResultSize = 1, 0
    dist = 0

    def bfsMapper(x):
        # x: (current, (dist, adjacent-list))
        d, adjlist = x[1]
        return [(adj, d+1) for adj in adjlist]

    while True:
        resultSize = newResultSize
        # Join the frontier with the graph to get all the adjacent nodes
        # of the frontier in current iteration. Then map to get the adjacent nodes.
        frontier = frontier.join(graph) \
                            .flatMap(bfsMapper).distinct()
                            
        result = result.union(frontier).reduceByKey(lambda d1, d2: min(d1, d2))
        # enforce lazy-evaluation
        newResultSize = result.count()
        if newResultSize == resultSize:
            break
        dist += 1

    return result, dist
  
