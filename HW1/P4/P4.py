import csv


import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()


def get_edges(tup):
    """
    For (issue, list_of_heroes), returns a list with edges between all heroes, with no self-edges.
    E.g.: (issue, [hero1, hero2, hero3]) would return:
    [(hero1, hero2), (hero1, hero3), (hero2, hero1), (hero2, hero3), (hero3, hero1), (hero3, hero2)]
    """
    issue, heroes = tup
    edges = []
    for i in xrange(len(heroes)):
        for j in xrange(len(heroes)):
            if i != j:
                edges.append((heroes[i], heroes[j]))
    return edges


def create_resolve_collisions():
    accum = sc.accumulator(0)

    def resolve_collisions(tup):
        """
        Resolves collision in combined vertices_dist RDD. Chooses the min distance
        and increments the accumulator if there is more than one distance.
        """
        key, distances = tup
        if len(distances) > 1:
            accum.add(1)
        return (key, min(distances))

    return accum, resolve_collisions


with open('source.csv', 'r') as infile:
    characters = sc.parallelize(list(csv.reader(infile))).map(lambda x: (x[0].strip(), x[1].strip()))

adjacency_list = characters.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).flatMap(get_edges).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)


def bfs(source):
    vertices_dist = sc.parallelize([(source, 0)])
    dist = 1
    current_level_size = -1
    accum = sc.accumulator(0)

    # stop if the current level size is the same as the number of intersections
    # (stored in accumulator)
    while current_level_size != accum.value:
        accum, resolve_collisions = create_resolve_collisions()

        # get list of current-level vertices
        current_vertices = set(vertices_dist.filter(lambda x: x[1] == dist - 1).map(lambda x: x[0]).collect())

        # get the neighbors of all current-level vertices
        neighbors = adjacency_list.filter(lambda x: x[0] in current_vertices).flatMap(lambda x: [(n, dist) for n in x[1]])
        current_level_size = neighbors.count()

        vertices_dist = vertices_dist.union(neighbors).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y).map(resolve_collisions)

        dist += 1

    return vertices_dist
