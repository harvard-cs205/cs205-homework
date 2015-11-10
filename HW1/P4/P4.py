# Marvel Graphs

import csv
from P4_bfs import *


def makegraph():

    data = csv.reader(open('source.csv','r'),delimiter=",")

    # make dictionary by comic
    graph = {}

    for line in data:
        if line[1] in graph:
            graph[line[1]] += [line[0]]
        else:
            graph[line[1]] = [line[0]]


    # make dictionary by person
    people_graph = {}

    # go through each person in each world level
    for people in graph.values():
        for person in people:

            if person not in people_graph:
                people_graph[person] = people[:]
                people_graph[person].remove(person)

            else:
                people_graph[person] += people[:]
                people_graph[person].remove(person)

    # make into RDD
    graph = sc.parallelize(people_graph.items())
    #graph = graph.flatMapValues(lambda x: x)

    return graph

marvelgraph = makegraph()

doc = open('captainamerica.txt','w')
doc.write(str(ssbfs(marvelgraph,'CAPTAIN AMERICA').collect()))
doc.close()
