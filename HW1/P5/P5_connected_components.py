def connected_components(graph):
    context = graph.context
    result = []
    while not graph.isEmpty():
        visited = context.emptyRDD()
        start = graph.first()
        toVisit = graph.filter(lambda KV: KV == start)
        while not toVisit.isEmpty():
            #Update record of previously visited nodes
            graph = graph.subtract(toVisit, 8)
            visited = visited.union(toVisit)
            #Update record of nodes to visit
            neighbors = toVisit.flatMap(lambda KV: KV[1]).collect()
            toVisit = graph.filter(lambda KV: KV[0] in neighbors)
        result.append(visited.count())
    return result
