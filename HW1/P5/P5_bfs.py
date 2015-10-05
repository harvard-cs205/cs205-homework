from pyspark import SparkContext, AccumulatorParam

class AccumNodes(AccumulatorParam):
    def addInPlace(self, val1, val2):
        val1 |= val2
        return val1

    def zero(self, val):
        return val

def parse_text(line):
    data = line.split(': ')
    return (int(data[0]), set(map(int, data[1].split())))

def I(x):
    return x

def swapkv((k, v)):
    return v, k

def bfs(sc, links, names, root, target):
    # add indices to names for future lookups
    name_index = names.zipWithIndex().map(lambda (n, i): (n, i+1))
    index_name = name_index.map(swapkv).sortByKey()
    root = name_index.lookup(root)[0]
    target = name_index.lookup(target)[0]
    traversed = set()
    current = set([root])
    # save paths from root to target as a rdd (child, parent)
    # initialize to (root, -1)
    paths = sc.parallelize([(root, -1)])

    while current:
        traversed |= current
        if target not in traversed: 
            filtered = links.filter(lambda (k, v): k in current)
            counter = sc.accumulator(set(), AccumNodes())
            filtered.values().foreach(lambda x: counter.add(x))
            # path from current nodes to children
            path = filtered.flatMapValues(I).map(swapkv)
            # to preserve directionality of the links, eliminate path from child to parent
            # drop paths with traversed nodes as children 
            paths = path.subtractByKey(paths).union(paths)
            current = counter.value - traversed
        else: # target reached
            break
    
    if target in traversed:
        paths = paths.repartition(32).sortByKey().cache()
        # trace path from target to root
        reverse_path = [target]
        next = paths.lookup(reverse_path[-1])[0]
        while next != -1:
            reverse_path.append(next)
            next = paths.lookup(reverse_path[-1])[0]
        return [index_name.lookup(i)[0] for i in reverse_path[::-1]]
    else:
        print "No path from", root, "to", target

if __name__ == '__main__':
    # set up partitioning params
    NPART = 9*16

    # initialize spark
    sc = SparkContext("local", appName="BFS")

    # load files
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', NPART).map(parse_text).cache()
    names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', NPART).cache()
    # links = sc.textFile('links-simple-sorted.txt', NPART).map(parse_text).cache()
    # names = sc.textFile('titles-sorted.txt', NPART).cache()
    
    # run BFS
    kevin_to_harvard = bfs(sc, links, names, 'Kevin_Bacon', 'Harvard_University')
    harvard_to_kevin = bfs(sc, links, names, 'Harvard_University', 'Kevin_Bacon')
    print kevin_to_harvard
    print harvard_to_kevin