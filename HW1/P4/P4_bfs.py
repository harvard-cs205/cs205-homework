from pyspark import SparkContext, AccumulatorParam

def bfs_serial(graph, root, iteration=10):
    '''
    Serial implementation of BFS
    Assuming diameter is 10
    '''
    paths = {}
    current = {root}
    for step in range(0, iteration+1):
        next = set()
        filtered = graph.filter(lambda (k, v): k in current).cache()
        for n in current:
            if n not in paths:
                paths[n] = step
                next |= filtered.lookup(n)[0]
        current = next - set(paths.keys())

    # return results
    numtraversed = len(paths) - 1 # excl. root
    return numtraversed

class AccumNodes(AccumulatorParam):
    def addInPlace(self, v1, v2):
        v1 |= v2
        return v1

    def zero(self, val):
        return val

def bfs_parallel(sc, graph, root):
    '''
    Parallel implementation
    '''
    traversed = set()
    current = {root}

    while current:
        filtered = graph.filter(lambda (k, v): k in current)
        traversed |= current
        counter = sc.accumulator(set(), AccumNodes())
        filtered.values().foreach(lambda x: counter.add(x))
        current = counter.value - traversed

    numtraversed = len(traversed) - 1 # excl. root
    return numtraversed
