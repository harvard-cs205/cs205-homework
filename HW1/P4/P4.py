from operator import add

# import findspark
# findspark.init('/home/lhoang/spark')

# import pyspark
# sc = pyspark.SparkContext(appName="spark1")


def create_graph(fileName):
    """
    Create a graph of character relationship given a file.
    :param fileName: The csv file to load where each line = (hero, comic).
    """
    src = sc.textFile(fileName)

    # convert records to (comic, hero)
    vk = src.map(lambda x: x.split('"')).map(lambda tup: (tup[3], tup[1]))

    # join records by comic to find which heroes are related
    g = vk.join(vk).map(lambda kv: sorted(list(set(list(kv[1]))))).filter(
        lambda seq: len(seq) > 1)

    # remove duplicate records, then make relationship symmetric
    # by creating reverse links: (hero1, hero2) to (hero2, hero1)
    gr = g.map(lambda seq: (seq[0], seq[1])).distinct().flatMap(
        lambda kv: [kv, (kv[1], kv[0])])

    # now reduce the list of all (hero_i, hero_j) records by hero_i
    # this gives a list of (hero_i, ([neighbors of hero_i], 0))
    # where distances are initialized to 0
    graph = gr.map(lambda kvp: (kvp[0], [kvp[1]])).reduceByKey(add).map(
            lambda kv: (kv[0], (list(set(kv[1])), -1)))

    return graph


def bfs_search_iteration(i, graph, root):
    """
    Perform breadth-first search at iteration i.
    :param i: The iteration.
    :param graph: The current graph.
    :param root: The root nodes to whose neighbors will update distances.
    """
    print 'i = ' + repr(i) + ', root count = ' + repr(len(root))

    neighbors = graph.filter(lambda kv: kv[0] in root).flatMap(
        lambda kv: kv[1][0]).collect()
    neighbors = list(set(neighbors))

    # increase distance for all neighbor nodes (excluding root)
    # if the distance isn't already set
    graph = graph.map(lambda kv: (kv[0], (kv[1][0], i)) if (
        (kv[0] in neighbors) and (kv[1][1] == -1)) else kv)

    return graph, neighbors


def bfs_search(graph, hero):
    """
    Perform breadth-first search to find shortest path from hero.
    :param hero: The root node to find paths from.
    """
    # initially root nodes only contains the source node
    # but will grow to contain more nodes at each iteration
    root = [hero]
    diameter = 10

    # initially set distance to self to zero
    graph = graph.map(lambda kv: (kv[0], (kv[1][0], 0)) if kv[
        0] == hero else kv)

    # iteratively find distances for other nodes
    for i in range(1, diameter + 1):
        graph, root = bfs_search_iteration(i, graph, root)

    return graph


def bfs_search_strange_behavior(graph, hero):
    """
    Similar to bfs_search but very strange behavior.
    The iteration below does not update the graph properly so every
    new iteration applies transformation to the original graph instead.
    """
    # initially root nodes only contains the source node
    # but will grow to contain more nodes at each iteration
    root = [hero]
    diameter = 10

    # initially set distance to self to zero
    graph = graph.map(lambda kv: (kv[0], (kv[1][0], 0)) if kv[
        0] == hero else kv)

    for i in range(1, diameter + 1):
        print 'i = ' + repr(i) + ', root count = ' + repr(len(root))

        neighbors = graph.filter(lambda kv: kv[0] in root).flatMap(
            lambda kv: kv[1][0]).collect()
        neighbors = list(set(neighbors))

        # increase distance for all neighbor nodes (excluding root)
        # if the distance isn't already set
        graph = graph.map(lambda kv: (kv[0], (kv[1][0], i)) if (
            (kv[0] in neighbors) and (kv[1][1] == -1)) else kv)

        root = neighbors

    return graph
