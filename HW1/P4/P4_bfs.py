# Perform breadth first search starting at src_name in an adjacency list
# rdd.  Returns an rdd which maps every node connected to src_name to its
# distance from src_name.
def bfs(adj_list_rdd, src_name, sc):
    accum = sc.accumulator(0)
    adj_list_rdd.cache()
    distances = sc.parallelize([(src_name, 0)])
    i = 0
    curr = 0
    c = adj_list_rdd.count()
    last = -1
    while accum.value != last:
        last = accum.value
        accum.value = 0
        neighbors_rdd = distances.filter(lambda x: x[1] == curr).join(adj_list_rdd) \
                         .flatMap(lambda (key, (num, l)): [(a, curr + 1) for a in l])
        distances = distances.union(neighbors_rdd).reduceByKey(lambda a, b: min(a,b)).cache()
        curr += 1
        distances.foreach(lambda x: accum.add(1))
    return distances