import time

num_of_partitions = 2

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def SSBFS( source, graph, accum ): 
    graph = graph.map( lambda ( x, y ): ( x, ( -1, y ) ) if x != source else ( x, ( 0, y ) ), preservesPartitioning = True )
    graph.cache()
    i = 0
    def adder( (x, y) ):
        accum.add(1)
        return (x, y[1][0])
    while( True ):
        source = graph.filter( lambda (x,y): y[0] == i ).mapValues( lambda y: y[1] ).flatMapValues( lambda y: y )
        assert copartitioned( graph, source )
        accum_before = accum.value
        not_visited = graph.filter( lambda (x,y): y[0] == -1 ).partitionBy(num_of_partitions)
        source = source.map( lambda (x,y) : (y, 1) ).distinct().join( not_visited ).map( lambda (x,y): adder( ( x, y ) ) ).partitionBy(num_of_partitions)
        source.cache()
        print source.count()
        accum_after = accum.value
        if accum_before == accum_after:
            break
        assert copartitioned( graph, source )
        graph = graph.leftOuterJoin( source ).mapValues( lambda y: ( ( i + 1, y[0][1] ), [] ) if y[1] is not None and y[0][0] == -1 else y ).mapValues( lambda y: y[0] ).partitionBy(num_of_partitions)
        graph.cache()
        i = i + 1
    return graph       