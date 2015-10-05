def update_child_links(node, dist, parent, neighbors, level):
    '''
    Filter nodes whose distances are the current level and return
    all links of child nodes with updated distance
    Otherwises, return the original one
    '''
    if dist == level:
        result = [(node, (dist, parent, neighbors))]
        result = result + [(neighbor, (level+1, node, [])) for neighbor in neighbors]
        return result
    else:
        return [(node, (dist, parent, neighbors))]

def update_graph(x, y):
    '''
    Choose the node with the minimum distance while keeping all
    neighbors of each node
    '''
    dist_x = x[0]
    dist_y = y[0]
    parent_x = x[1]
    parent_y = y[1]
    neighbors_x = x[2]
    neighbors_y = y[2]

    if neighbors_x == []:
        neighbors = neighbors_y
    elif neighbors_y == []:
        neighbors = neighbors_x
    elif (neighbors_x == []) and (neighbors_y == []):
        neighbors = []

    if dist_x < dist_y:
        return (dist_x, parent_x,neighbors)
    else:
        return (dist_y, parent_y,neighbors)

def BFS_with_short_path(rdd_graph, start_node, end_node):
    '''
    Given a (K, V) RDD with K = node and V = [connecting nodes],
    find the shortest paths from a source node to all other nodes in a graph.
    '''
    # rdd_graph => (node, (distance, parent, neighbors)) RDD
    rdd_graph = rdd_graph.map(lambda (node, neighbors): (node, (0, None, neighbors)) if (node == start_node) else (node, (float('inf'), None, neighbors)))
    level = 0
    while True:
      # flatMap graph to RDD 
      rdd_graph = rdd_graph.flatMap(lambda (node, (dist, parent, neighbors)) : update_child_links(node, dist, parent, neighbors, level))
      rdd_graph = rdd_graph.reduceByKey(lambda x, y: update_graph(x,y))

      # if distance of end_node is not infinite number
      if rdd_graph.map(lambda x: x).lookup(end_node)[0][0] != float('inf'):
        break
      level += 1

    level += 1
    # Return updated RDD graph and level
    return (rdd_graph, level)

def Generate_Short_Path(rdd_graph, rdd_pages, start_node, end_node):
    '''
    Generate short path from start_node to end_node given a Graph
    '''
    short_path = [end_node]
    find_node = end_node 
    # Starting with end_node, trace back all the way to start_node
    # to find a short path
    while True:
      parent_node = rdd_graph.map(lambda x: x).lookup(find_node)[0][1]
      if (parent_node == None) and (parent_node == start_node):
        short_path.append(start_node)
        break
      short_path.append(parent_node)
      find_node = parent_node

    # Convert short path using labels to one using page titles
    pages_for_short_path = []
    short_path = short_path[::-1]
    for i in range(len(short_path)):
      page = rdd_pages.map(lambda x : x).lookup(short_path[i])[0]
      pages_for_short_path.append(page)

    # Path (Using Labels, Using Page Titles)
    return (" -> ".join(short_path), " -> ".join(pages_for_short_path))

