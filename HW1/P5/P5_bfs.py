from pyspark import SparkContext
from pyspark import AccumulatorParam

class SetAccumulator(AccumulatorParam):
    # hash of nod to parent
    def zero(self, initialValue):
        return {}

    # combine the dictionaries
    def addInPlace(self, s1, s2):
        union = s2.copy()
        union.update(s1)
        return union

# update parent pointers so we can backtrack
def setParents(child, parent):
    pointer = {}
    for c in child:
        pointer[c] = parent
    return pointer

def bfs(sc, graph, source, dest):
    # initialize variables
    visited = sc.accumulator({source: None}, SetAccumulator())
    frontier = set([source])
    distance = 0
    found = False

    # terminate if found nothing new or found dest
    while len(frontier) and not found:
        # update values
        distance += 1
        pre_vals = set(visited.value.keys())

        # iterate through current level and update
        graph.filter(lambda (x, y): x in frontier) \
             .foreach(lambda (x, y): visited.add(setParents(y, x)))
        
        # update frontier
        frontier = set(visited.value.keys()) - pre_vals
        
        # check if we've reached the dest node
        if dest in visited.value.keys(): 
            found = True
    
    pointer = visited.value

    # create path from dest to source
    path = [dest]
    while dest != source:
        path.append(pointer[dest])
        dest = pointer[dest]

    # reverse path so it goes forwards
    return path[::-1]

def generate_path(sc, graph, start_name, end_name):
    # get name1 id
    name1 = page_names.filter(lambda (k, v): v == start_name)
    assert name1.count() == 1
    name1_id = name1.collect()[0][0]

    # get name2 id
    name2 = page_names.filter(lambda (k, v): v == end_name)
    assert name2.count() == 1
    name2_id = name2.collect()[0][0]

    # get bfs path
    path = bfs(sc, graph, name1_id, name2_id)

    # create name path
    names = [start_name]
    for node in path[1:]:
        name = page_names.filter(lambda (k, v): k == node).collect()[0][1]
        names.append(name)
    
    # turn into string
    return " -> ".join(names)

# constructs links as node with edges
def construct_node(line):
    src, dests = line.split(': ')
    dests = set([int(to) for to in dests.split(' ')])
    return (int(src), dests)

if __name__ == '__main__':
    sc = SparkContext()
    sc.setLogLevel('WARN')

    # load data
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

    # construct graph
    graph = links.map(construct_node)
    graph = graph.partitionBy(64).cache()

    # create page_names rdd with id as key
    page_names = page_names.zipWithIndex().map(lambda (n, id): (id+1, n))
    page_names = page_names.sortByKey().cache()

    # print out paths
    print generate_path(sc, graph, "Kevin_Bacon", "Harvard_University")
    print generate_path(sc, graph, "Harvard_University", "Kevin_Bacon")
