from pyspark import SparkContext, AccumulatorParam

class AccumNodes(AccumulatorParam):
    def addInPlace(self, value1, value2):
        value1 |= value2
        return value1

    def zero(self, value):
        return value

def parse_text(line):
    data = line.split(': ')
    return (int(data[0]), set([int(i) for i in data[1].split()]))

def bfs_parallel(sc, links, names, root, target):
    # convert names to indices
    names_indices = names.zipWithIndex().map(lambda (n, i): (n, i+1))
    indices_names = names_indices.map(lambda (n, i): (i, n))
    root = names_indices.lookup(root)[0]
    target = names_indices.lookup(target)[0]

    traversed = set()
    current = {root}
    # save paths from root to target
    paths = sc.parallelize([(root, -1)])

    while current:
        traversed |= current
        if target in traversed: # target reached
            break
        filtered = links.filter(lambda (k, v): k in current)
        counter = sc.accumulator(set(), AccumNodes())
        filtered.values().foreach(lambda x: counter.add(x))
        # get path from current nodes to children
        path = filtered.flatMapValues(lambda x: x).map(
            lambda (parent, child): (child, parent))
        # to preserve directionality of the links
        # we need to eliminate path from child to parent
        # so we drop path that has a traversed node as child 
        paths = path.subtractByKey(paths).union(paths)
        current = counter.value - traversed
        
    if target in traversed:
        # trace path
        reverse_path = [target]
        while True:
            next = paths.lookup(reverse_path[-1])[0]
            if next == -1:
                break
            reverse_path.append(next)
        print [indices_names.lookup(i)[0] for i in reverse_path[::-1]]
    else:
        print "No path between %s and %s" %(root, target)

if __name__=='__main__':
    NPART = 32
    # initialize spark
    sc = SparkContext()
    links = sc.textFile(
        's3://Harvard-CS205/wikipedia/links-simple-sorted.txt', NPART)
    names = sc.textFile(
        's3://Harvard-CS205/wikipedia/titles-sorted.txt', NPART)

    # links = sc.textFile(
    #     'links-simple-sorted.txt', NPART)
    # names = sc.textFile(
    #     'titles-sorted.txt', NPART)

    # load data into a rdd (issue, character)
    links = links.map(parse_text).cache()
    bfs_parallel(sc, links, names, 'Kevin_Bacon', 'Harvard_University')
    # bfs_parallel(sc, links, names, 'Jeff', 'Stephen')