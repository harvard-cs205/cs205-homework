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

def electsmallest(p):# map: (4, {1,2,3,4}) => (1, {1,2,3,4})
    return min(p[1]), p[1]

def reverseEdge(p):# flatMap: (a, {b,c}) => (b,a), (c,a)
    for s in p[1]:
        yield s, set( [p[0]] )

def dedupEdge(p):# flatMap: (3, {1,4}) => ((1,3),1), ((3,4),1)
    for s in p[1]:
        if p[0]<s:
            yield (p[0], s), 1
        else:
            yield (s, p[0]), 1

def reputEdge(p):
    if p[1] == 2:# flatMap: ((1,9), 2) => (1,{9}), (9,{1})
        yield p[0][0], set([ p[0][1] ])
        yield p[0][1], set([ p[0][0] ])
    elif p[1] == 1:# flatMap: ((4,5), 1) => (4,{}), (5,{})
        yield p[0][0], set()
        yield p[0][1], set()

def buildEdge(p):
    for s in p[1]:
        if s > p[0]:
            yield s, p[0]

def exploreCom(p):
    for s in p[1]:
        yield s, p[1]

def unionreduce(v1, v2):
    return v1 | v2

def components(graph_rdd):
    component_union = graph_rdd.map( unionNeighbors ).reduceByKey( unionreduce ) #union node with its neighbors, choose the smallest node as key
    before = component_union.count()
    print "iteration", 0
   #print component_union.collect()
    i = 0
    while True:
        component_union = component_union.flatMap( exploreCom ).reduceByKey( unionreduce ).map( electsmallest ).reduceByKey( unionreduce )
        #print component_union.collect()
        i += 1
        print "iteration", i
        after = component_union.count()
        if after == before:
            break
        before = after
        
    return component_union.count(), max(  component_union.map( lambda p: len(p[1]) ).collect()  )

if __name__ == '__main__':
    sc = SparkContext("local", appName="Spark1")
    sc.setLogLevel('WARN')
    
    print "load text"
    #txtfile = sc.textFile('component.txt', 32) #load text
    txtfile = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    print "build graph"
    adjacent_graph = txtfile.map(link_string_to_KV) #transform to graph
    
    # add symmetric edges
    print "symmetric"
    symmetric = adjacent_graph.flatMap( reverseEdge ).union(adjacent_graph).reduceByKey( unionreduce ) # add reversed edges
    print "built symmetric graph"
    #print symmetric.collect()
    
    num_com = components(symmetric)
    print "symmetric has component", num_com[0], "largest component", num_com[1]
    


    # remove unidirectional edges
    print "=================================================================="
    print "bidirectional"
    bidirectional = adjacent_graph.flatMap( dedupEdge ).reduceByKey( lambda v1, v2: v1+v2 ) # edge rdd with smaller nodes in the front, the bidirectional edges will have duplicate
    bidirectional = bidirectional.flatMap( reputEdge ).reduceByKey( unionreduce )
    print "built bidirectional graph"
    #print bidirectional.collect()
    
    num_com = components(bidirectional)
    print "bidirectional has component", num_com[0], "largest component", num_com[1]
    