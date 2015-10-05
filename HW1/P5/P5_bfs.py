import pyspark

#HELPER FUNCTIONS

#Parse single line in input file and return a (k,v) tuple
def construct_neighbor_graph(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

############################################################
# My approach in P4 takes too long to converge, so insteat I
# tried a simple naive way by directly implementing bfs
# search using set and it performs amazingly well...
############################################################

#Search for all path from start to goal
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
def shortest_path(graph, start, goal):
    try:
        return next(bfs_paths(graph, start, goal))
    except StopIteration:
        return None



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
h_k_shortest_path = shortest_path(neighbor_graph_simple, Harvard_University, Kevin_Bacon)
h_k_shortest_path = sc.parallelize(h_k_shortest_path).map(lambda x: page_names.lookup(x)[0]).collect()
print 'Shortest Path from “Harvard University” to "Kevin Bacon": \n', h_k_shortest_path


print 'Begin searching for Shortest Path from "Kevin Bacon" to "Harvard University": \n'
k_h_shortest_path = shortest_path(neighbor_graph_simple, Kevin_Bacon, Harvard_University)
k_h_shortest_path = sc.parallelize(k_h_shortest_path).map(lambda x: page_names.lookup(x)[0]).collect()
print 'Shortest Path from "Kevin Bacon” to “Harvard University": \n', k_h_shortest_path

