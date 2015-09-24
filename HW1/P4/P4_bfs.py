#My graph is represented as an RDD of key-value pairs. The key is the name of the comic book character as found in the input file and the value is a set of the names of all other comic book characters who share at least one issue with that character, as the names appear in the input file.

def bfs(graph, start):
    #Used to generate accumulator and empty RDD
    context = graph.context
    accum = context.accumulator(0)
    distances = context.emptyRDD()
    #toVisit is the set of neighbor nodes that have not yet been visited
    toVisit = {start}
    while len(toVisit) != 0:
        #Accumulator value must be retrieved outside of a task
        distance = accum.value
        neighbors = graph.filter(lambda KV: KV[0] in toVisit)
        new_distances = neighbors.map(lambda KV: (KV[0], distance))
        #Update record of previously visited nodes
        distances = distances.union(new_distances).reduceByKey(lambda x, y: min(x, y))
        visited = distances.keys()
        #Update record of nodes to visit
        toVisit = neighbors.flatMap(lambda KV: KV[1]).subtract(visited).distinct().collect()
        accum.add(1)
    return distances
