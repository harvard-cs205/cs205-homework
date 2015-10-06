from pyspark import SparkContext
from pyspark import AccumulatorParam

# set accumulator to keep track of visited nodes
class SetAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return set()
    def addInPlace(self, v1, v2):
        return v1 | v2

def rdd_bfs(root, graph, sc):
    visited = sc.accumulator(set(), SetAccumulatorParam());
    neigh_set = set([root])
    while neigh_set:
        curr_visited = visited.value
        # not doing any calls to collect, only using the accumulator
        graph.filter(lambda (k, v): k in neigh_set).foreach(lambda (k, v): visited.add(set(v)))
        neigh_set = visited.value - curr_visited
    # only need nodes touched, not distances
    return visited.value
