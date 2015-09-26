def bfs_parents(graph, start):
    context = graph.context
    iteration = context.accumulator(0)
    data = context.emptyRDD()
    toVisit = graph.filter(lambda KV: KV[0] == start).map(lambda KV: (KV[0], (KV[1], '')))
    toVisitList = {}
    while toVisit.count() != 0:
        distance = iteration.value
        new_data = toVisit.map(lambda KV: (KV[0], (distance, KV[1][1])))
        data = data.union(new_data).reduceByKey(lambda x, y: min(x, y))
        pairs = toVisit.map(lambda KV: [(x, KV[0]) for x in KV[1][0]])
        neighbors = pairs.flatMap(lambda KV: KV)
        toVisit = graph.join(neighbors).subtractByKey(data)
        iteration.add(1)
    return data
    
def shortest_path(data, end):
    sorted_data = data.sortByKey()
    path = []
    next = end
    while next != '':
        node = sorted_data.lookup(next)
        if len(node) > 0:
            path.append((next, node[0][0]))
            next = node[0][1]
            if len(node) > 1:
                path.append('ERROR: NON-UNIQUE RESULT')
        else:
            next = ''
    path.reverse()
    return path
