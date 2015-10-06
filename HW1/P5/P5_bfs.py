import pyspark
import partial
sc = pyspark.SparkContext()

#Parse single line in input file and return a (k,v) tuple
def construct_neighbor_graph(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

def reduce_nodes(nodes_list):
    nodes_list = list(nodes_list)
    return sorted(nodes_list, key = lambda x: x[1])[0]

def update_graph(node, mc_graph, accum):
    # extract values from node for clarity
    node = list((node[0], list(node[1])))
    character_name = node[0]
    adj_list = node[1][0]
    dist = node[1][1]
    is_used = node[1][2]
    result = []
    if dist < float("inf"):
        for i in adj_list:
            new_node = [i, list(mc_graph[i])]
            if new_node[1][1] > (dist + 1):
                new_node[1][1] = (dist + 1)
                result.append(new_node)
        node[1][2] = True
        accum.add(1)
    result.append(node)
    return result

def BFS(MC_Graph, init_node):
    MC_Graph_bfs = MC_Graph.map(lambda (x, y): (x, (y, 0, False))
                                if (x == init_node)
                                else (x, (y, float("inf"), False)))
    tmp = dict(MC_Graph_bfs.collect())
    accum = sc.accumulator(0)
    iterate_flag = True
    iteration_count = 1
    while iterate_flag:
        updated_nodes = MC_Graph_bfs.filter(lambda x:
            x[1][2] == False).flatMap(
            partial(update_graph,
                mc_graph = dict(MC_Graph_bfs.collect()),
                    accum = accum)).groupByKey().mapValues(reduce_nodes)

        MC_Graph_bfs = MC_Graph_bfs.filter(lambda x: x[1][2] == True).union(updated_nodes)

        if (accum.value == 0):
            iterate_flag = False
        else:
            print 'In iteration ', iteration_count, ', ', accum.value, 'nodes discovered;\n'
            accum = sc.accumulator(0)
            iteration_count = iteration_count + 1

    result = MC_Graph_bfs.filter(lambda x: x[1][1] < float("inf")).map(lambda (x, y): (x, (y[0], y[1])))
    return result


#Search for all shortest path from start to goal
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

#Search for the shortest path
def shortest_paths(graph, start, goal):
    dist_graph = BFS(graph, start)
    goal_distance =  dist_graph.filter(lambda (K, V): K == goal).collect()
    goal_distance = goal_distance[1][1]

    if goal_distance == float("inf"):
        print goal, ' is not connected to ', start, ', please verify input.\n'
        return
    else:
        return bfs_paths(dist_graph, start, goal)


if __name__ == '__main__':

    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

    # construct and cache neighbor_graph and page_names (for lookup purpose)
    neighbor_graph = links.map(construct_neighbor_graph).partitionBy(64).cache()
    page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n)).sortByKey().cache()

    # neighbor_graph_simple is used for simple bfs search for shortest path
    neighbor_graph_simple = neighbor_graph.map(lambda x: dict({x[0]: set(x[1])})).collect()

    # find Kevin Bacon
    Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
    Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

    # find Harvard University
    Harvard_University = page_names.filter(lambda (K, V):
                                   V == 'Harvard_University').collect()
    # This should be [(node_id, 'Harvard_University')]
    Harvard_University = Harvard_University[0][0]  # extract node id

    print 'Begin searching for Shortest Path from "Harvard University" to "Kevin Bacon": \n'
    h_k_shortest_path = shortest_paths(neighbor_graph_simple, Harvard_University, Kevin_Bacon)
    h_k_shortest_path = sc.parallelize(h_k_shortest_path).map(lambda x: page_names.lookup(x)[0]).collect()
    print 'Shortest Path from “Harvard University” to "Kevin Bacon": \n', h_k_shortest_path


    print 'Begin searching for Shortest Path from "Kevin Bacon" to "Harvard University": \n'
    k_h_shortest_path = shortest_paths(neighbor_graph_simple, Kevin_Bacon, Harvard_University)
    k_h_shortest_path = sc.parallelize(k_h_shortest_path).map(lambda x: page_names.lookup(x)[0]).collect()
    print 'Shortest Path from "Kevin Bacon” to “Harvard University": \n', k_h_shortest_path

