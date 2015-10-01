import csv


import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()


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
    characters = sc.parallelize(list(csv.reader(infile))).map(lambda x: (x[0].strip(), x[1].strip()))

adjacency_list = characters.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).flatMap(get_edges).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)

# map onto joined RDD w/ adjacency list to get the right neighbors list
get_neighbors = lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][1]))

# reduce by key with this function to or the flag, min the distance, and take the larger neighbors list
collapse_nodes = lambda x, y: (any((x[0], y[0])), min(x[1], y[1]), larger(x[2], y[2]))


def bfs(source):
    accum, update_accumulator = create_update_accumulator()
    vertices = sc.parallelize([(source, (False, 0, []))]).join(adjacency_list).map(get_neighbors)
    dist = 0
    old_num_falses = -1
    accum = sc.accumulator(0)

    # stop if number of non-explored nodes has not changed since last iteration
    # (stored in accumulator)
    while dist < 5:
        old_num_falses = accum.value
        vertices = vertices.flatMap(lambda x: [(neighbor, (False, dist + 1, [])) for neighbor in x[1][2]] + [(x[0], (True, x[1][1], x[1][2]))]).reduceByKey(collapse_nodes)
        to_search = vertices.filter(lambda x: x[1][0] is False).join(adjacency_list).map(get_neighbors)
        already_searched = vertices.filter(lambda x: x[1][0] is True)
        vertices = to_search.union(already_searched).reduceByKey(collapse_nodes).map(update_accumulator)
        dist += 1

    return vertices

a = bfs("ORWELL").collect()
