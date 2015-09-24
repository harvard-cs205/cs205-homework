def bfs(graph, start):
    context = graph.context
    accum = context.accumulator(0)
    toVisit = {start}
    #Arbitrary filter to get empty RDD
    distances = context.emptyRDD()
    while len(toVisit) != 0:
        neighbors = graph.filter(lambda KV: KV[0] in toVisit)
        distance = accum.value
        new_distances = neighbors.map(lambda KV: (KV[0], distance))
        distances = distances.union(new_distances).reduceByKey(lambda x, y: min(x, y))
        accum.add(1)
        visited = distances.keys()
        toVisit = neighbors.flatMap(lambda KV: KV[1]).subtract(visited).collect()
    return distances
