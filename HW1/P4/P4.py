from P4_bfs import *

if __name__ == "__main__":
    N = 16  # Number of partitions

    sc = pyspark.SparkContext("local[4]")  # Let's take it easy with my 2009 MacBook
    source_rdd = sc.textFile('source.csv', minPartitions=N)

    # Getting our data into shape:
    tuple_rdd = source_rdd.map(lambda entry: tuple(entry.split(u'","')), True).map(
        lambda tup: (tup[1].replace(u'"', u''), tup[0].replace(u'"', u'')), True)  # [(comic-book, hero), ...]

    # Generate edges between vertices:
    edges_rdd = tuple_rdd.join(tuple_rdd, numPartitions=N).map(lambda (k, v): v, True)  # preserve only edge pairs
    edges_rdd = edges_rdd.distinct(N).filter(lambda (hero1, hero2): hero1 != hero2)  # filter duplicates, links to self

    # Now group by character for easy retrieval:
    graph = edges_rdd.groupByKey(numPartitions=N).mapValues(list).sortByKey(numPartitions=N)
    graph.cache()

    #print graph.lookup('CAPTAIN AMERICA')
    rdd = distance_to_all_nodes_from("CAPTAIN AMERICA", graph)

    print rdd.take(20)


