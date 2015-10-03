def process_line(s):
    'Convert line from data file to (comic, [character]) pair'
    line_split = s.split('","')
    return (line_split[1].replace('"',''),[line_split[0].replace('"','')])

def current_count(i):
    return (lambda x: i)

def ss_bfs(sc, graph, origin):
    '''
    Single-Source Breadth-First Search.

    sc: spark context
    graph: each node is assumed to consist of a (string, set) tuple, where the
           string represents the name of a character and the set contains the
           names of all characters that co-appear with that character
    origin: string, the name of a character to use as origin for the SS-BFS
    '''
    to_touch = sc.parallelize([(origin, 0)])
    touched = sc.parallelize([(origin, 0)])

    dist_so_far = 1
    # while the next level queue is not empty
    while not to_touch.isEmpty():

        # get all the neighbors of all characters at current distance
        all_neighbors = to_touch.join(graph).flatMap(lambda (k,(v1,v2)): [(v,dist_so_far) for v in v2])

        # throw out duplicates and ones we've already touched
        to_touch = all_neighbors.reduceByKey(lambda d1, d2: d1).subtractByKey(touched).cache()

        # save new distances
        touched = touched.union(to_touch).cache()
        dist_so_far += 1

    return touched
