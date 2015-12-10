# import findspark
# findspark.init()
import pyspark
import numpy as np

sc = pyspark.SparkContext()

N = 32

def link_string_to_KV(s):
        src, dests = s.split(': ')
        dests = [int(to) for to in dests.split(' ')]
        return (int(src), dests)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def symmetric_graph(graph):
    
    lonely = graph.filter(lambda x: x[1] == [])
    a = graph.map(lambda x: [(x[0], i) for i in x[1]]).flatMap(lambda x: x)
    inverted = graph.map(lambda x: [(i,x[0]) for i in x[1]])
    b  = inverted.flatMap(lambda x: x)
    symmetric_graph = a.intersection(b).groupByKey().map(lambda x: (x[0], list(x[1]))).union(lonely)
    symmetric_graph = symmetric_graph.union(graph.subtractByKey(symmetric_graph).map(lambda x: (x[0], [])))
    
    return symmetric_graph

def filterer(i):
    def filt(x):
        return x==itr_acc.value
    return filt

def get_first_non_empty_cogroup(x2):
    for elem in x2:
        elem = list(elem)
        if len(elem) > 0:
            return elem[0]

def BFS(graph, starting_node, N):
    
        #We set the starting node to have distance = 0 
    nodes = sc.parallelize([(starting_node, 1)]).partitionBy(N)
    
    graph.partitionBy(N).cache()
    
    total_char_acc = sc.accumulator(1)
    it_acc = sc.accumulator(0)

    while True:
        it_acc += 1
        dist = it_acc.value

        nodes_2 = nodes.filter(lambda x: filterer(x[1])).join(graph).flatMap(lambda x: x[1][1]).distinct().map(lambda x: (x, dist + 1)).partitionBy(N)

        nodes = nodes.cogroup(nodes_2).map(lambda x: (x[0], get_first_non_empty_cogroup(list(x[1])))).partitionBy(N).cache()
        
        assert copartitioned(nodes_2, nodes)

        if nodes.count() == total_char_acc.value:
            break

        total_char_acc += nodes.count() - total_char_acc.value
        
    return nodes.count(), nodes



def biggest_component(list_components):
    u, indices = np.unique(list_components, return_inverse=True)
    biggest = u[np.argmax(np.bincount(indices))]
    return list_components.count(biggest)

def connected_components(graph, N):
    
    components = sc.parallelize([])
    n_components = sc.accumulator(0)
    n_nodes = graph.count()
    counted_nodes = 0
    
    while counted_nodes != n_nodes:
    
        component_index = n_components.value
        
        target_node = graph.keys().takeSample(False, 1, int(np.random.uniform(0,100000,1)))
                
        [nodes_added, touched_nodes] = BFS(graph, target_node[0], N)
                
        graph = graph.subtractByKey(touched_nodes).partitionBy(N).cache()
                
        touched_nodes = touched_nodes.map(lambda x: (x, component_index)).partitionBy(N)
        
        components = components.union(touched_nodes).partitionBy(N).cache()
        
        assert copartitioned(touched_nodes, components)
        
        n_components += 1
        counted_nodes = nodes_added + counted_nodes  
        print counted_nodes
        
    return biggest_component(components.map(lambda x: x[1]).collect()),  n_components.value


if __name__ == '__main__':


    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

    # links = sc.textFile('./testdata.txt')
    # page_names = sc.textFile('./pages.txt')
    # process links into (node #, [neighbor node #, neighbor node #, ...]

    neighbor_graph = links.map(link_string_to_KV)

    [maxC, nC] = connected_components(neighbor_graph, N)

    neighbor_symmetric = symmetric_graph(neighbor_graph)

    [maxC2, nC2] = connected_components(neighbor_symmetric, N)

    print "For the graph where each link is taken as symmetric, the total number of connected components is", nC, "and the biggest component has:", maxC, "elements."

    print "For the graph where two pages are only considered linked if they link to each other, the total number of connected components is", nC2, "and the biggest component has:", maxC2, "elements."



