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
    edge_rdd = build_edges_rdd(links, N)
    lookup_table = page_names.zipWithIndex().mapValues(lambda v: v + 1)  # 1-indexed

    # Distance between nodes in network
    # distance_to = distance_between("Kevin_Bacon", "Harvard_University", edge_rdd, lookup_table, N).collect()
    # print distance_to
    # distance_from = distance_between("Harvard_University", "Kevin_Bacon", edge_rdd, lookup_table, N).collect()
    # print distance_from

    # Make new graphs
    mirrored_rdd = edge_rdd.map(lambda (k, v): (v, k))  # helper graph
    undirected_rdd = edge_rdd.union(mirrored_rdd).distinct(N)
    bi_link_rdd = edge_rdd.join(mirrored_rdd).partitionBy(N).filter(lambda (k, (v, w)): v == w)\
        .map(lambda (k, (v, w)): (k, v)).distinct(N)

    # Connected components
    components_un = connected_components(undirected_rdd, N)
    components_bi = connected_components(bi_link_rdd, N)
    print "Components of the undirected graph ", components_un
    print "Components of bi-link graph: ", components_bi
