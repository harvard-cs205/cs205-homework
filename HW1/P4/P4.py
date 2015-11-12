from P4_bfs import *

# Parameters
partitions = 4

# (K, V) is (issue, Character)
comics = sc.textFile('source.csv').map(lambda line: (line.split('"')[3],
                                       line.split('"')[1]))

# Join version
edges = comics.join(comics).map(lambda x: (x[1][0],
                                x[1][1])).filter(lambda x: x[0] != x[1])
graph = edges.groupByKey().map(lambda x: (x[0], set(list(x[1]))))\
    .partitionBy(partitions).cache()

# Test and computation of the number of touched nodes

# Version2
roots = ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']
for root in roots:
    %time distance = ss_bfs2(graph, root)
    num_node_visited = distance.count()
    # Substract the root
    print('{} nodes visited for the character {}'.format(num_node_visited - 1,
          root))

# Version3
roots = ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']
# Initialization
# Adding a distance in the entry graph
bfs_graph = graph.mapValues(lambda v: (float('inf'), v))
for root in roots:
    %time bfs = ss_bfs3(bfs_graph, root)
    distance = bfs.filter(lambda x: x[1][0] < float('inf'))
    num_node_visited = distance.count()
    # Substract the root
    print('{} nodes visited for the character {}'.format(num_node_visited - 1,
          root))
