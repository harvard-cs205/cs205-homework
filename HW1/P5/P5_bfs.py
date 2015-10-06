from pyspark import SparkContext
from pyspark import AccumulatorParam

# set accumulator to keep track of visited nodes
# will only update if have not seen that node before
class DictAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return initialValue
    def addInPlace(self, v1, v2):
        # TODO
        for n in v2.keys():
            if n not in v1.keys():
                v1[n] = v2[n]
        return v1

# converts list to dictionary, to pass to accumulator
def list_to_dict(k, d):
    temp = {}
    for elt in set(d):
        temp[elt] = k
    return temp

def rdd_bfs(root, target, graph, sc):
    prev = sc.accumulator({root: -1}, DictAccumulatorParam())
    neigh_set = set([root])
    while neigh_set:
        curr_prev = set(prev.value.keys())
        # not doing any calls to collect, only using the accumulator
        graph.filter(lambda (k, v): k in neigh_set).foreach(lambda (k, v): prev.add(list_to_dict(k, v)))
        # update to new nodes on the current frontier
        neigh_set = set(prev.value.keys()) - curr_prev
        # short circuit if found target node
        for n in neigh_set:
            if n == target:
                return prev.value
    # if not found, return empty dictionary
    return {}
