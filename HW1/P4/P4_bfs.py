__author__ = 'xiaowen'

def bfs(node, graph):
    # initialize graph with node as vertice and initialize distance
    bfs_rdd = graph.map(lambda (x, y): (x, (0, y)) if (x == node) else (x, (float("inf"), y)))
    distances = bfs_rdd.map(lambda (x, y): (x, y[0])).collectAsMap()
    count_nodes = 0
    counter = graph.context.accumulator(0)
    while counter.value == 0:
        bfs_rdd = bfs_rdd.map(lambda x: update_distance(x, distances))
        distances = bfs_rdd.map(lambda (n, (d, nb)): (n, d)).collectAsMap()
        prev = count_nodes
        count_nodes = bfs_rdd.filter(lambda (v, (d, n)): d < float("inf")).count()
        if prev == count_nodes:
            counter.add(1)
    return count_nodes


def update_distance(x, distances):
    node = x[0]
    dis = x[1][0]
    neighbors = x[1][1]
    if dis == float("inf"):
        # distances of neighbors
        n_dist = [distances[i] for i in neighbors]
        # get the min
        min_d = min(n_dist) + 1
        # update my distance with min + 1 if it is smaller
        if min_d < dis:
            dis = min_d
    return node, (dis, neighbors)
