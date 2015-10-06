import pyspark
sc = pyspark.SparkContext()

sc.setLogLevel('WARN')


def parse_links(x):
    """
    Takes arg of form "source: n1 n2 n3 n4 n5" and returns
    (source, [n1, n2, n3, n4, n5])
    """
    source, neighbors = x.split(': ')
    source = int(source)
    neighbors = [int(i) for i in neighbors.split(' ')]

    return (source, neighbors)


links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
links = links.map(parse_links)
index_to_title = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt').zipWithIndex().map(lambda x: (x[1] + 1, x[0])).cache()
title_to_index = index_to_title.map(lambda x: (x[1], x[0])).cache()


def larger(x, y):
    """
    Return larger of two lists. If tie, returns x.
    For use in add_neighbors.
    """
    if len(y) > len(x):
        return y
    else:
        return x


def collapse_nodes(x, y):
    """
    Reducer that takes:
        * Max of the flags
        * Min of the distances
        * Larger of the neighbors lists
        * Previous of the node with the smaller distance
    """
    if x[3] is None:
        previous = y[3]
    elif y[3] is None:
        previous = x[3]
    elif x[1] <= y[1]:
        previous = x[3]
    else:
        previous = y[3]

    return (max((x[0], y[0])), min(x[1], y[1]), larger(x[2], y[2]), previous)


def bfs(source, target):
    """
    Performs breadth-first search starting at source and ending once
    target is found or all connected nodes have been touched.

    Flags mean:
    * 0 = node has not yet been reached
    * 1 = node has been reached, but neighbors have not been marked as reached yet
    * 2 = node has been reached and neighbors have been marked as reached

    Vertices are of form (node, (flag, dist, [neighbors], previous))
    """

    def initialize_nodes(x):
        # initialize flag to 0
        flag = 0

        # initialize distance to a high number
        dist = 9999

        # initialize the source's flag to 1 and distance to 0
        if x[0] == source:
            flag = 1
            dist = 0

        return (x[0], (flag, dist, x[1], None))

    vertices = links.map(initialize_nodes).cache()

    def create_add_neighbors():
        """
        Closure to create the mapper function with
        accumulators in the proper scope.
        """

        found_target = sc.accumulator(0)
        accum = sc.accumulator(0)

        def add_neighbors(node):
            """
            Mapper that "activates" neighbors of nodes found in the previous
            iteration.
            """

            (node, (flag, dist, neighbors, previous)) = node
            nodes = []
            if flag == 1:

                # increment accumulator
                accum.add(1)

                # if we've found the target, increment
                # the found_target accumulator to stop
                # the search
                if node == target:
                    found_target.add(1)

                # add neighbors to RDD
                for neighbor in neighbors:
                    nodes.append((neighbor, (1, dist + 1, [], node)))
                nodes.append((node, (2, dist, neighbors, previous)))
            else:
                nodes.append((node, (flag, dist, neighbors, previous)))

            return nodes

        return accum, found_target, add_neighbors

    accum, found_target, add_neighbors = create_add_neighbors()
    dist = 0
    old_num_ones = -1

    while accum.value != old_num_ones and found_target.value == 0:
        accum, found_target, add_neighbors = create_add_neighbors()
        vertices = vertices.flatMap(add_neighbors).reduceByKey(collapse_nodes)
        vertices.count()
        dist += 1

    # return only nodes that we've touched and explored
    return vertices.filter(lambda x: x[1][0] == 2)


def find_path(source, target, vertices=None, path=None):
    """
    Searches through the BFS results recursively, from the
    target back to the source, and returns a list ordered
    from source to target.
    """
    if vertices is None:
        source = title_to_index.lookup(source)[0]
        target = title_to_index.lookup(target)[0]
        vertices = dict(bfs(source, target).collect())
        path = [target]

    next_node = vertices[target][3]
    path.append(next_node)
    if next_node == source:
        return [index_to_title.lookup(x)[0] for x in path[::-1]]
    else:
        return find_path(source, next_node, vertices, path)

print find_path("Harvard_University", "Kevin_Bacon")
print find_path("Kevin_Bacon", "Harvard_University")
