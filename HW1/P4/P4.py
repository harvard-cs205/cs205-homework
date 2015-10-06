import csv


import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

sc.setLogLevel('WARN')


def larger(x, y):
    """
    Return larger of two lists. If tie, returns x.
    """
    if len(y) > len(x):
        return y
    else:
        return x


def get_edges(tup):
    """
    For (issue, list_of_heroes), returns a list with edges between all heroes, with no self-edges.
    E.g.: (issue, [hero1, hero2, hero3]) would return:
    [(hero1, hero2), (hero1, hero3), (hero2, hero1), (hero2, hero3), (hero3, hero1), (hero3, hero2)]
    """
    issue, heroes = tup
    edges = set()
    for i in xrange(len(heroes)):
        for j in xrange(len(heroes)):
            if i != j:
                edges.add((heroes[i], heroes[j]))
    return edges


def create_update_accumulator():
    accum = sc.accumulator(0)

    def update_accumulator(tup):
        """
        Adds to accumulator if sees a False flag
        """
        key, (flag, dist, neighbors) = tup
        if flag is False:
            accum.add(1)
        return tup

    return accum, update_accumulator


with open('source.csv', 'r') as infile:
    characters = sc.parallelize(list(csv.reader(infile))).map(lambda x: (x[0], x[1]))

# reduce by key with this function to max the flag, min the distance, and take the larger neighbors list
collapse_nodes = lambda x, y: (max((x[0], y[0])), min(x[1], y[1]), larger(x[2], y[2]))


def bfs(source):
    def initialize_nodes(x):
        flag = 0
        dist = 9999
        if x[0] == source:
            flag = 1
            dist = 0
        return (x[0], (flag, dist, x[1]))

    def create_add_neighbors():
        accum = sc.accumulator(0)

        def add_neighbors(node):
            (node, (flag, dist, neighbors)) = node
            nodes = []
            if flag == 1:
                accum.add(1)
                for neighbor in neighbors:
                    nodes.append((neighbor, (1, dist + 1, [])))
                nodes.append((node, (2, dist, neighbors)))
            else:
                nodes.append((node, (flag, dist, neighbors)))

            return nodes

        return accum, add_neighbors

    vertices = characters.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).flatMap(get_edges).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y).map(initialize_nodes).cache()
    accum, add_neighbors = create_add_neighbors()
    # ("Node", (flag, dist, [neighbors]))
    dist = 0
    old_num_ones = -1

    # stop if number of non-explored nodes has not changed since last iteration
    # (stored in accumulator)
    # Flags:
    # 0 = have not touched
    # 1 = touched, need to add neighbors
    # 2 = have added all neighbors
    while accum.value != old_num_ones:
        old_num_ones = accum.value
        accum, add_neighbors = create_add_neighbors()
        vertices = vertices.flatMap(add_neighbors).reduceByKey(collapse_nodes)
        vertices.count()
        dist += 1

    return vertices.filter(lambda x: x[1][0] == 2)

a = bfs("CAPTAIN AMERICA")
