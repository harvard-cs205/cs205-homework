from P4_bfs import *


if __name__ == "__main__":
    N = 32  # Number of partitions
    sc = pyspark.SparkContext("local[8]")
    sc.setLogLevel("ERROR")

    # Take source
    source_rdd = sc.textFile('source.csv', N, False)

    # Getting our data into shape:
    tuple_rdd = source_rdd.map(lambda entry: tuple(entry.split('","')), True).map(
        lambda tup: (tup[1].replace('"', ''), tup[0].replace('"', '')), True)  # [(comic-book, hero), ...]

    # Generate edges between nodes:
    edges_rdd = tuple_rdd.join(tuple_rdd, numPartitions=N).map(lambda (k, v): v, True)  # preserve only edge pairs
    edges_rdd = edges_rdd.union(edges_rdd.map(lambda (k, v): (v, k), True))  # undirected graph
    edges_rdd = edges_rdd.distinct(N).filter(lambda (hero1, hero2): hero1 != hero2)  # filter duplicates, links to self

    # Now group by character for easy retrieval:
    graph = edges_rdd.groupByKey(numPartitions=N).mapValues(list).sortByKey(numPartitions=N)
    graph.cache()

    print "DONE BUILDING GRAPH. STARTING BFS..."

    # Distance to nodes in network (Choose...)
    # one = distance_to_all_nodes_serial("ORWELL", graph).collect()  # SERIAL VERSION
    two = distance_to_all_nodes_edge("CAPTAIN AMERICA", edges_rdd, N).collect()  # EDGE TUPLE VERSION (preferred)
    # three = distance_to_all_nodes_spark("ORWELL", graph).collect()  # GRAPH-LIST VERSION
    print two
    print "Nodes: ", len(two)
