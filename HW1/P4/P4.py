from operator import add

from p4_bfs import *

import csv


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


with open('source.csv', 'r') as infile:
    characters = sc.parallelize(list(csv.reader(infile))).map(lambda x: (x[0], x[1]))

vertices = characters.map(lambda x: (x[1], [x[0]])).reduceByKey(add).flatMap(get_edges).map(lambda x: (x[0], [x[1]])).reduceByKey(add)

a = bfs(vertices, "CAPTAIN AMERICA")
