from functools import partial

#########################################################################
# BFS function
# Takes in a graph RDD and a character name, update the graph iteratively
# using Breadth-First Search algorithm. This is the last version that uses 
# Spark Accumulators and rdd.filters() for optimization
#########################################################################

def reduce_nodes(nodes_list):
    nodes_list = list(nodes_list)
    return sorted(nodes_list, key = lambda x: x[1])[0]

#########################################################################
# 
# The update_graph is performed on each node in the filtered range (unused)
# during each iteration. 
#
#########################################################################

def update_graph(node, mc_graph, accum):
    # extract values from node for clarity
    character_name = node[0]
    adj_list = node[1][0]
    dist = node[1][1]
    is_used = node[1][2]
    
    mc_graph = mc_graph.value
    result = []

    #if the node has been touched (dist < inf), use it to touch its adjacent nodes
    if dist < float("inf"):
        for i in adj_list:
            #Get the adjacent node
            new_node = (i, mc_graph[i])

            # Set distance if it is current min and append it to result list
            if new_node[1][1] > (dist + 1):
                new_node[1][1] = (dist + 1)
                result.append(new_node)
        # Set the is_used state to True since it has already touch its adj_list
        node[1][2] = True
        accum.add(1)
    # Append the node itself as an element of result
    result.append(node)
    return result


def BFS(MC_Graph, init_node):

    # Transform MC_Graph (character_name, [adj_list]) to MC_Graph_bfs to 
    # perform BFS search, MC_Graph_bfs is in the following form:
    # (character_name, ([adj_list], distance, is_used))
    #
    # is_used:Bool value, stores whether the node has been use to activate 
    # other nodes for distance calculation, used in RDD.filter()
    #
    # the distance for every node, except start node, is set to infinite as 
    # initial state

    MC_Graph_bfs = MC_Graph.map(lambda (x, y): (x, (y, 0, False)) 
                                if (x == init_node) 
                                else (x, (y, float("inf"), False)))

    # Convert MC_Graph_bfs to adictionary and broadcast it every worker
    # This is used for lookup during each update
    tmp = dict(MC_Graph_bfs.collect())
    broadcast_graph = sc.broadcast(tmp)

    # Accumulator set to 0
    accum = sc.accumulator(0) 
    iterate_flag = True
    # Perform BFS iteratively. Each iteration expands the search frontier by one hop.
    while iterate_flag:    

        #########################################################################
        # This step first use rdd.filter() to optimize the amount of data transferred 
        # during each update by only look into nodes which haven't been used to update
        # other nodes (nodes used for update must already have their min distance set)
        #
        # The update is done by calling update_graph graph function (see below), and reduce 
        # the result to get minimum distance for each single node 
        #########################################################################

        updated_nodes = MC_Graph_bfs.filter(lambda x: 
            x[1][2] == False).flatMap(
            partial(update_graph, 
                mc_graph = broadcast_graph, accum = accum)).groupByKey().mapValues(reduce_nodes)
                
        # Since only a part of MC_Graph_bfs is updated each time, union the used and 
        # unused nodes to get the most up-to-date gragh
        MC_Graph_bfs = MC_Graph_bfs.filter(lambda x: x[1][2] == True).union(updated_nodes)

        #Stop when no new nodes are found in current iteration
        if (accum.value == 0):
            iterate_flag = False
        else:
            accum = sc.accumulator(0)
      
    # get the number of touched nodes by counting the number of nodes with distance 
    # smaller than infinite  
    touched_nodes_num = MC_Graph_bfs.filter(lambda x: x[1][1] < float("inf")).count()

    return touched_nodes_num
    