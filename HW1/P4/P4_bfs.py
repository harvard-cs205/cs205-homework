# Helper function - updates accumulator for newly explored nodes
def nodes_update(x, explored):
    explored.add(1)
    return x

# BFSS
def bfs(sc, charGraph, startNode):
    
    # Explored depth of character graph
    dist = 0

    # Initial search nodes = starting node
    searchNodes = sc.parallelize([startNode]).map(lambda x: (x, 0))

    # Initial frontier = starting node
    frontier = sc.parallelize([startNode]).map(lambda x: (x, 0))

    # Accumulator to keep track of explored nodes
    explored = sc.accumulator(1)
    
    # Keep going until there are no further nodes to expand
    while not searchNodes.isEmpty():
        
        # Increment depth
        dist += 1

        # Look up all characters that are touched by all characters in the frontier
        children = charGraph.join(searchNodes).flatMap(lambda x: x[1][0]).map(lambda x: (x, dist)).partitionBy(20)

        # Combine the new characters with the frontier
        # Drop any characters that are already present with a lower distance
        frontier = frontier.union(children).reduceByKey(min).partitionBy(20).cache()

        # Filter out the nodes that characters that need to be checked in the next iteration
        # Update accumulator for newly explored nodes
        searchNodes = frontier.filter(lambda x: x[1]==dist).map(lambda x: nodes_update(x, explored)).partitionBy(20).cache()
        
    # Return accumulator value
    return explored.value
