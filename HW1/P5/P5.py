from P5_bfs import *
from P5_connected_components import *

if __name__ == '__main__':

    N = 40  # Number of partitions
    sc = pyspark.SparkContext()  # "local[24]"
    sc.setLogLevel("ERROR")

    # Get files
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', N, use_unicode=False)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', N, use_unicode=False)

    # Build RDDs
    edge_rdd = build_edges_rdd(links)
    print edge_rdd.take(10)
    lookup_table = page_names.zipWithIndex().mapValues(lambda v: v + 1)  # 1-indexed
    print lookup_table.lookup("Kevin_Bacon")

    # Distance between nodes in network
    distance = distance_between("Kevin_Bacon", "Harvard_University", edge_rdd, lookup_table, N).collect()
    print distance

    # Connected components
