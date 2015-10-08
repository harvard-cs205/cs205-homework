from P5_bfs import *
#import findspark
#findspark.init()
import pyspark

def preprocess_links( links ):
    links_processed = links.map( lambda x: ( x.split( ":" )[0], x.split( ":")[1] ) )#.sample(True, 0.001)
    links_processed.count()
    links_processed.cache()
    links_processed_final = links_processed.map( lambda (x,y): ( x, y.split() ) )
    links_processed_final.count()
    links_processed_final.cache()
    return links_processed_final


def preprocess_titles( titles, start_end_pairs ):
    titles_index = titles.zipWithIndex().map( lambda (x,y): (x, y+1))
    titles_index.count()
    titles_index.cache()
    pair_dict = []
    for pair in start_end_pairs:
        start_index = titles_index.lookup( pair[0] ) #print titles_index.lookup( "Kevin_Bacon") [2729536]
        end_index = titles_index.lookup( pair[1] ) #[2152782]
        pair_dict = pair_dict + [ { "start_index": str(start_index[0]), "end_index": str(end_index[0]) } ]
    return pair_dict

    
def find_path( graph_with_distance, distance, final_node ):
    nodes_to_be_found = [ final_node ]
    path = []
    for i in xrange( distance, -1, -1 ):
        nodes_to_be_found_temp = []
        graph_with_distance_i = graph_with_distance.filter( lambda (x,y): y[0] == i )
        graph_with_distance_i.count()
        graph_with_distance_i.cache()
        for node in nodes_to_be_found:
            nodes_to_be_found_temp = nodes_to_be_found_temp + map( lambda (x,y): x, graph_with_distance_i.filter( lambda (x,y): True if node in y[1] else False ).collect() )
        path = path + [ ( i+1, nodes_to_be_found ) ]
        nodes_to_be_found = nodes_to_be_found_temp
    return path
        
number_of_partitions = 32

def main():
    start = time.time()
    sc = pyspark.SparkContext()
    accum = sc.accumulator(0)
    
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
    #links = sc.textFile( "links-simple-sorted.txt" )
    #titles = sc.textFile( "titles-sorted.txt" )
    
    #graph = preprocess_data( "./source.csv", sc )
    graph = preprocess_links( links )
    graph = graph.partitionBy( number_of_partitions )
    graph.count()
    graph.cache()
    
    #start_nodes = [ { "start_index": "CAPTAIN AMERICA", "end_index": "BLAIR, CALEY" } ]
    start_end_pairs = [ ( "Kevin_Bacon", "Harvard_University" ), ( "Kevin_Bacon", "Harvard_University" ) ] 
    start_nodes = preprocess_titles( page_names, start_end_pairs )
    print "OUTPUT--OUTPUT--OUTPUT--OUTPUT"
    print start_nodes
    
    for i in start_nodes:
        print i
        ( output, distance ) = SSBFS( i[ "start_index" ], graph, i[ "end_index" ]  )
        print "OUTPUT--OUTPUT--OUTPUT--OUTPUT"
        print output.count()
        graph_with_distance = output.filter( lambda (x,y): y[0] != -1 )
        print "OUTPUT--OUTPUT--OUTPUT--OUTPUT"
        print graph_with_distance.count()
        path = find_path( graph_with_distance, distance, i[ "end_index" ] )
        path.append( ( 0, [ i[ "start_index" ] ] ) )
        print "OUTPUT--OUTPUT--OUTPUT--OUTPUT"
        print i
        print path
        
    print time.time() - start
    
if __name__ == "__main__": main()