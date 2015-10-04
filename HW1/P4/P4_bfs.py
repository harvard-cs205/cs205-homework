def process_line(s):
    'Convert line from data file to (comic, [character]) pair'
    line_split = s.split('","')
    return (line_split[1].replace('"',''),[line_split[0].replace('"','')])

def to_touch_untouched(to_touch):
    return lambda (k,(d,ns)): k in to_touch and d == -1

def ss_bfs(sc, graph, origin):
    '''
    Single-Source Breadth-First Search.

    sc: spark context
    graph: each node is assumed to consist of a (string, set) tuple, where the
           string represents the name of a character and the set contains the
           names of all characters that co-appear with that character
    origin: string, the name of a character to use as origin for the SS-BFS
    '''
    to_touch = set([origin])
    ttrdd = sc.parallelize(to_touch)
    touched = sc.emptyRDD()

    # set all distances to infinity (i.e., -1)
    graph_dists = graph.mapValues(lambda v: (-1, v))

    dist_so_far = 0
    # imitating a do-while loop
    while True:
        acc = sc.accumulator(0)

        # find untouched nodes from our list of nodes to touch
        ttns = graph_dists.filter(to_touch_untouched(to_touch))

        # update set of neighbors to touch
        def count_get_neighbors(ns):
            acc.add(1)
            return ns
        ttrdd = ttns.flatMap(lambda (k,(d,ns)): count_get_neighbors(ns))
        to_touch = set(ttrdd.collect())

        # set new distances
        touched = touched.union(ttns.mapValues(lambda v: dist_so_far)).cache()
        graph_dists = graph_dists.leftOuterJoin(ttns).mapValues(lambda (v,w): (dist_so_far, w[1]) if w else v).cache()

        dist_so_far += 1

        # the do-while condition
        if acc.value == 0:
            break

    return touched
