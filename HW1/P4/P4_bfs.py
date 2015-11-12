from pyspark import AccumulatorParam

def BFS(start, rdd, sc):
    class SetAccumulatorParam(AccumulatorParam):
        def zero(self, initialValue):
            return set()

        def addInPlace(self, v1, v2):
            v1 |= v2
            return v1
    visited = sc.accumulator(set(), SetAccumulatorParam())
    #visited = set()
    not_visited = set([start])
    iteration = 0
    
    while(len(not_visited)>0):
        iteration += 1
        pre = set(list(visited.value))
        rdd.filter(lambda (K, V): K in not_visited).foreach(lambda (K, V): visited.add(V))
        print iteration, start, len(visited.value), len(pre)
        not_visited = visited.value-pre

    return len(visited.value)-1, iteration
