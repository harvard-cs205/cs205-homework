#My graph is represented as an RDD of key-value pairs. The key is the name of the comic book character as found in the input file and the value is a set of the names of all other comic book characters who share at least one issue with that character, as the names appear in the input file.

def bfs(graph, start):
    #Used to generate accumulator and empty RDD
    context = graph.context
    iteration = context.accumulator(0)
    touched = context.accumulator(0)
    distances = context.emptyRDD()
    #toVisit is the set of neighbor nodes that have not yet been visited
    toVisit = graph.filter(lambda KV: KV[0] == start)
    while toVisit.count() != 0:
        toVisit.foreach(lambda _: touched.add(1))
        #Accumulator value must be retrieved outside of a task
        distance = iteration.value
        new_distances = toVisit.map(lambda KV: (KV[0], distance))
        #Update record of previously visited nodes
        distances = distances.union(new_distances).reduceByKey(lambda x, y: min(x, y))
        #Update record of nodes to visit
        neighbors = toVisit.flatMap(lambda KV: KV[1]).collect()
        toVisit = graph.filter(lambda KV: KV[0] in neighbors).subtractByKey(distances)
        iteration.add(1)
    return touched.value
