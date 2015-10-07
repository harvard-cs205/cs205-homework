from P4_bfs import *
import findspark
findspark.init()
import pyspark

def preprocess_data( filename, sc ):
    textFile = sc.textFile( filename )
    textFile_pre_process = textFile.map( lambda x: ( x.split( "\"" )[3], x.split( "\"" )[1] ) ).partitionBy(num_of_partitions)
    issue_char_mapping = textFile_pre_process.mapValues( lambda y: [y] ).reduceByKey( lambda a, b: a + b, numPartitions = num_of_partitions )
    assert copartitioned( textFile_pre_process, issue_char_mapping )
    char_issue_join = textFile_pre_process.join( issue_char_mapping ).partitionBy(num_of_partitions)
    char_adjacency_graph = char_issue_join.map( lambda ( x, y ): y ).reduceByKey( lambda x, y: x + y ).map( lambda ( x, y ): ( x, list( set( [q for q in y if q != x] ) ) ), preservesPartitioning = True ) 
    graph = char_adjacency_graph
    graph.cache()
    return graph


def main():
    start = time.time()
    sc = pyspark.SparkContext()
    accum = sc.accumulator(0)
    graph = preprocess_data( "./source.csv", sc )
    
    for i in [ "MISS THING/MARY", "ORWELL", "CAPTAIN AMERICA" ]:
        output = SSBFS( i, graph, accum )
        print output.count()
        print output.filter( lambda (x,y): y[0] == -1 ).count()
        print i
    print time.time() - start
    
if __name__ == "__main__": main()