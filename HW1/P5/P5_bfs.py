def process_line(s):
    'Convert line from data file to (comic, [character]) pair'
    line_split = s.split('","')
    return (line_split[1].replace('"',''),[line_split[0].replace('"','')])

def process_links(s):
    'Convert line from data file to adjacency list'
    line_split = s.split(': ')
    return (int(line_split[0]), [int(n) for n in line_split[1].split(' ')])

def get_by_key(k, rdd_dict):
    return rdd_dict.lookup(k)[0]

def to_touch_untouched(to_touch):
    return (lambda (k,ns): k in to_touch)

def bfs_path(sc, graph, origin, destination):
    '''
    Returns shortest directed path from origin to destination, or None if
    origin and destination are not connected in graph.

    sc: spark context
    graph: each node is assumed to consist of a (string, set) tuple, where the
           string represents the name of a character and the set contains the
           names of all characters that co-appear with that character
    origin: key of origin node
    destination: key of destination node
    '''
    to_touch = set([origin])
    ttrdd = sc.parallelize([(origin, None)])
    touched = sc.emptyRDD()

    # accumulator to keep track of whether we found destination
    found = sc.accumulator(0)

    # keep going until we find our destination or run out of graph
    while not ttrdd.isEmpty() and found.value == 0:
        print 'hi from {0} to {1}'.format(str(origin),str(destination))

        # find untouched nodes from our list of nodes to touch
        ttns = graph.filter(to_touch_untouched(to_touch))

        # update set of neighbors to touch and check if we're at destination
        def neighbors_and_check_dest(k, ns):
            if k == destination:
                found.add(1)
            return [(n,k) for n in ns]
        ttrdd = ttns.flatMap(lambda x: neighbors_and_check_dest(*x)).subtractByKey(touched)
        to_touch = set(ttrdd.keys().collect())

        # set previous nodes
        touched = touched.union(ttrdd)

    # trace destination path backwards
    path = [destination]
    current = destination
    while current != origin:
        lst = touched.lookup(current)
        if lst:
            current = lst[0]
        else:
            break
        path.append(current)

    return path[::-1] if current and current != destination else None
