# import findspark
# findspark.init()
import pyspark
sc = pyspark.SparkContext()
N = 32

def link_string_to_KV(s):
        src, dests = s.split(': ')
        dests = [int(to) for to in dests.split(' ')]
        return (int(src), dests)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def filterer(i):
    def filt(x):
        return x==itr_acc.value - 1
    return filt

def get_first_non_empty_cogroup(x2):
    
    for elem in x2:
        elem = list(elem)
        if len(elem) > 0:
            return elem[0]

def shortest_path(graph, nodeA, nodeB, N):
    
    graph = graph.partitionBy(N).cache()
    nodes = sc.parallelize([(nodeA, 0)]).partitionBy(N)
    node_destination = [nodeB]
    # a = a.map(lambda x: (x[0], (list(x[1][0]))[0], (list(x[1][1]))))

    total_char_acc = sc.accumulator(1)
    it_acc = sc.accumulator(0)
    nodes_4 = sc.parallelize([]).partitionBy(N)

    while True:
        
        it_acc += 1
        dist = it_acc.value
                
        assert copartitioned(nodes, graph)
        
        nodes_1 = nodes.filter(lambda x: filterer(x[1])).join(graph)
        
        assert copartitioned(nodes, nodes_1)
        
        nodes_2 = nodes_1.flatMap(lambda x: x[1][1]).distinct().map(lambda x: (x, dist))

        nodes_3 = nodes_1.map(lambda x: (x[0],x[1][1],dist))
        nodes_3 = nodes_3.map(lambda x: [(x[0], i, dist) for i in x[1]])
                
        nodes_4 = nodes_3.union(nodes_4)
        
        assert copartitioned(nodes_3, nodes_4)

        nodes = nodes.cogroup(nodes_2).map(lambda x: (x[0], get_first_non_empty_cogroup(list(x[1])))).partitionBy(N).cache()

        
        if nodes_2.map(lambda x: x[0]).filter(lambda x: node_destination[0] == x).count() > 0:
            print "Distance:", dist
            break


    path = [node_destination]
    node_destination = [(0,nodeB)]
    nodes_4 = nodes_4.flatMap(lambda x: x)

    for i in range(dist,0,-1):
        
        node_destination = nodes_4.filter(lambda x: x[2] == i and x[1] in [y for _,y in node_destination]).map(lambda x: (x[1], x[0])).collect()
        path.append(node_destination)

    return dist, path


def transformToWords(b):
    path = []
    for i in b:
        if type(i[0]) is int:
            path.append(page_names.lookup(i[0])[0])
        else:
            for j in i:           
                path.append([page_names.lookup(k)[0] for k in j])
    return path

if __name__ == '__main__':


    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

    #links = sc.textFile('./testdata.txt')
    #page_names = sc.textFile('./pages.txt')
    # process links into (node #, [neighbor node #, neighbor node #, ...]

    neighbor_graph = links.map(link_string_to_KV)

    page_numbers = page_names.zipWithIndex().map(lambda x:(x[0],x[1]+1))

    Kevin = page_numbers.lookup('Kevin_Bacon')[0]
    Harvard = page_numbers.lookup('Harvard_University')[0]

    page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
    page_names = page_names.sortByKey().cache()

    
    [distKH, pathKH] = shortest_path(neighbor_graph, Kevin, Harvard, N)

    [distHK, pathHK] = shortest_path(neighbor_graph,  Harvard, Kevin, N)

    print "The distance from Harvard University to Kevin Bacon is", distHK,"and the possible paths are:", transformToWords(pathHK),". The distance from Kevin Bacon to Harvard University is", distKH,"and the possible paths are:", transformToWords(pathKH)



    
