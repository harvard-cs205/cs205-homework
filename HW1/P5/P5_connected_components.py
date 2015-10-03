from pyspark import SparkContext
from pyspark import AccumulatorParam

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = set( int(x) for x in dests.split(' ') )
    return (int(src), dests)


def unionNeighbors(p):
    uset = p[1]
    uset.add(p[0])
    return min(uset), uset

def reverseEdge(p):
    for s in p[1]:
        yield s, set( [p[0]] )

def unionreduce(v1, v2):
    return v1 | v2

def components(graph_rdd):
    before = graph_rdd.count()
    component_union = graph_rdd.map( unionNeighbors ).reduceByKey( unionreduce ) #union node with its neighbors
    after = component_union.count()
    print component_union.collect()
    while after < before:
        component_union = component_union.map( unionNeighbors ).reduceByKey( unionreduce )
        before = after
        after = component_union.count()
        print component_union.collect()
        
    print after

if __name__ == '__main__':
    sc = SparkContext("local", appName="Spark1")
    sc.setLogLevel('WARN')
    
    txtfile = sc.textFile('component.txt', 32) #load text
    #txtfile = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    
    adjacent_graph = txtfile.map(link_string_to_KV) #transform to graph
    
    # add symmetric edges
    symmetric = adjacent_graph.flatMap( reverseEdge ).union(adjacent_graph).reduceByKey( unionreduce ) # add reversed edges
    print "symmetric"
    print symmetric.collect()
    components(symmetric)
    