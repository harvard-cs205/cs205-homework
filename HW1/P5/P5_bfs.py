from pyspark import AccumulatorParam

# Accumulator used for BFS. Basically the algorithm adds all the visited
# nodes in one iteration into this accumulator instance.
class SetAccm(AccumulatorParam):
    def zero(self, s):
        return s

    def addInPlace(self, s1, s2):
        s1 |= s2
        return s1

def bfs(graph, source):
    sc = graph.context
    childParent = sc.parallelize([(source, -1)])

    frontier, closed = set([source]), set([source])

    while True:
        # Initialize the accumulator in each iteration
        tmpFoundAccm = sc.accumulator(set(), SetAccm())
        # Those entries in frontier queue will be searched
        toSearch = graph.filter(lambda (k, _): k in frontier)
        # Add all the adjacent nodes into the accumulator
        toSearch.foreach(lambda (k, adj): tmpFoundAccm.add(set(adj)))

        frontier = tmpFoundAccm.value - closed
        if len(frontier) == 0:
            return None

        closed |= tmpFoundAccm.value
    return closed

def shortestPath(graph, source, end):
    sc = graph.context
    # RDD to record (childIndex, parentIndex), source's parent is -1
    childParent = sc.parallelize([(source, -1)])
    # frontier queue and close set
    frontier, closed = set([source]), set([source])

    while True:
        # Initialize the accumulator in each iteration
        tmpFoundAccm = sc.accumulator(set(), SetAccm())
        # Those entries in frontier queue will be searched
        toSearch = graph.filter(lambda (k, _): k in frontier)
        # Add all the adjacent nodes into the accumulator
        toSearch.foreach(lambda (k, adj): tmpFoundAccm.add(set(adj)))
        # Create child-parent pair
        curChildParent = toSearch.flatMap(lambda (p, chdr): [(c, p) for c in chdr])
        # Reduce the child-parent pair, choose any parent as the child's parent is acceptable
        curChildParent = curChildParent.reduceByKey(lambda p1, p2: p1)
        # Set only those pairs whose child has not been visited before.
        curChildParent = curChildParent.filter(lambda (c, p): c not in closed)
        childParent = childParent.union(curChildParent)

        frontier = tmpFoundAccm.value - closed
        if len(frontier) == 0:
            return None

        closed |= tmpFoundAccm.value
        
        if end in frontier:
            # At this point, a complete path from source to end is found.
            # chdToFind: current child whose parent needs to be found.
            dist, chdToFind = 0, end
            path = [chdToFind]

            while True:
                # Find the parent of the current chdToFind
                entry = childParent.filter(lambda (k, _): k == chdToFind).collect()
                if len(entry) == 0:
                    return None
                chdToFind = entry[0][1]
                if chdToFind == -1:
                    break

                path.append(chdToFind)
                dist += 1
            # Reverse the path so that it is source to end
            path = path[::-1]
            return path, dist
    return None

def connected(graph):
    keyList = graph.map(lambda kv: (kv[0]))
    count = 0
    while True:
        try:
            source = keyList.take(1)[0]
            closed = bfs(graph, source)
            count += 1
            # Remove those that are visited
            keyList = keyList.filter(lambda k: k not in closed)
            keyList.cache()

            print 'Count for this iter is {0}'.format(count)

        except Exception, e:
            print 'Exception occured, count is {0}'.format(count)
            print str(e)
            break
    return count