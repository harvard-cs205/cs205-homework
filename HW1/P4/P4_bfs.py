# Author: George Lok
# P4.py

import time

def edgeMap(x) :
    # Add counts for all entries
    # -1 means disconnected.
    return (x[0], set(x[1]), -1)

def createMapFunctor(count, nextNodes) :
    # Updates counts for unvisited nodes in the queue.
    def mapFunctor((k, s, c,)) :
        if k in nextNodes :
            if c < 0 :   
                return (k, s,count)
        return (k, s,c)
    return mapFunctor

def reduceFunctor((k1, s1, c1), (k2, s2, c2)) :
    # Used to merge the next set of nodes we will jump to
    return (0, s1 | s2, 0 )

def SSBFS(source, graph, sc) :
    numVisited = sc.accumulator(0) # Keeps track of how many nodes we've visted
    nodes = graph.map(edgeMap)
    # nextNodes contain the frontier that we will jump to the next iteration
    nextNodes = set([source])
    count = 0
    last = 0
    while True:
        print "Iteration " + str(count)
        last = numVisited.value
        nodes = nodes.map(createMapFunctor(count, nextNodes))

        def filterFunctor(x) :
            return x[2] == count

        changedNodes = nodes.filter(filterFunctor)

        # Using an accumulator based off problem spec
        # It's unnecessary given the structure of this solution, 
        # since the space is comparably small
        changedNodes.foreach(lambda x : numVisited.add(1))

        # Semantically, there are no additional nodes we have touched
        # Note that checking nextNodes would be the non-accumulator way
        # of doing this check.
        if last == numVisited.value :
            break

        # returns (0, nextNodes, 0)  
        reduceResult = changedNodes.reduce(reduceFunctor)

        nextNodes = reduceResult[1]
        count += 1

    # Returns number of nodes touched and a RDD containing shortest distance from source to each node.
    return numVisited.value, nodes