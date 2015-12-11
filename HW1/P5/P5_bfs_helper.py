import time

num_of_partitions = 32

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def SSBFS( source, graph, end ): 
    graph = graph.map( lambda ( x, y ): ( x, ( -1, y ) ) if x != source else ( x, ( 0, y ) ), preservesPartitioning = True )
    graph.cache()
    i = 0
    def adder( (x, y) ):
        return (x, y[1][0])
    while( True ):
        source = graph.filter( lambda (x,y): y[0] == i ).mapValues( lambda y: y[1] ).flatMapValues( lambda y: y )
        print "OUTPUT--OUTPUT--OUTPUT--OUTPUT"
        print source.take(5)
        print source.count()
        assert copartitioned( graph, source )
        not_visited = graph.filter( lambda (x,y): y[0] == -1 ).partitionBy(num_of_partitions)   
        # eliminate nodes already visited
        source = source.map( lambda (x,y) : (y, 1) ).distinct().join( not_visited ).map( lambda (x,y): adder( ( x, y ) ) ).partitionBy(num_of_partitions)
        source.cache()
        source_count = source.count()
        print "OUTPUT--OUTPUT--OUTPUT--OUTPUT"
        print source_count
        if len( source.lookup( end ) ) > 0 or source_count == 0:
            break
        assert copartitioned( graph, source )
        # find next set of source nodes
        graph = graph.leftOuterJoin( source ).mapValues( lambda y: ( ( i + 1, y[0][1] ), [] ) if y[1] is not None and y[0][0] == -1 else y ).mapValues( lambda y: y[0] ).partitionBy(num_of_partitions)
        graph.cache()
        #if len( source.lookup( end ) ) > 0:
        #    break
        i = i + 1
    return ( graph, i )