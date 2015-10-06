import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

sc.setLogLevel('WARN')


def larger(x, y):
    """
    Return larger of two lists. If tie, returns x.
    (For use in add_neighbors)
    """
    if len(y) > len(x):
        return y
    else:
        return x

# reduce by key with this function to max the flag, min the distance, and take the larger neighbors list
collapse_nodes = lambda x, y: (max((x[0], y[0])), min(x[1], y[1]), larger(x[2], y[2]))


def bfs(vertices, source):
    """
    Performs breadth-first search starting at source and ending once
    all connected nodes have been touched.

    Flags mean:
    * 0 = node has not yet been reached
    * 1 = node has been reached, but neighbors have not been marked as reached yet
    * 2 = node has been reached and neighbors have been marked as reached

    Vertices are of form (node, (flag, dist, [neighbors]))
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

        return (x[0], (flag, dist, x[1]))

    vertices = vertices.map(initialize_nodes).cache()

    def create_add_neighbors():
        """
        Closure to create mapper with accumulators
        in the proper scope.
        """

        accum = sc.accumulator(0)

        def add_neighbors(node):
            """
            Mapper that "activates" neighbors of nodes found in the previous
            iteration.
            """

            (node, (flag, dist, neighbors)) = node
            nodes = []

            if flag == 1:
                # increment accumulator
                accum.add(1)

                # add neighbors to RDD
                for neighbor in neighbors:
                    nodes.append((neighbor, (1, dist + 1, [])))
                nodes.append((node, (2, dist, neighbors)))
            else:
                nodes.append((node, (flag, dist, neighbors)))

            return nodes

        return accum, add_neighbors

    accum, add_neighbors = create_add_neighbors()
    dist = 0
    old_num_ones = -1

    while accum.value != old_num_ones:
        old_num_ones = accum.value
        accum, add_neighbors = create_add_neighbors()
        vertices = vertices.flatMap(add_neighbors).reduceByKey(collapse_nodes)
        vertices.count()
        dist += 1

    # return only vertices that we've touched and explored
    return vertices.filter(lambda x: x[1][0] == 2)
