import pyspark
from P5_bfs import *

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

if __name__ == '__main__':
    sc = pyspark.SparkContext()
    sc.setLogLevel('WARN')
    #links = sc.textFile('/Users/haosutang/links-simple-sorted.txt')
    #page_names = sc.textFile('/Users/haosutang/titles-sorted.txt')
    links = sc.textFile('/Users/haosutang/links-simple-sorted.txt')
    page_names = sc.textFile('/Users/haosutang/titles-sorted.txt')
    neighbor_graph = links.map(link_string_to_KV)
    page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
    page_names = page_names.sortByKey().cache()

    # find Kevin Bacon
    Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
    # This should be [(node_id, 'Kevin_Bacon')]
    assert len(Kevin_Bacon) == 1
    Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

    # find Harvard University
    Harvard_University = page_names.filter(lambda (K, V):
                                           V == 'Harvard_University').collect()
    # This should be [(node_id, 'Harvard_University')]
    assert len(Harvard_University) == 1
    Harvard_University = Harvard_University[0][0]  # extract node id

    neighbor = neighbor_graph.cache()

    shortest_path = BFS_shortest_path(neighbor,  Kevin_Bacon, Harvard_University)

    shortest_path_id = []

    for i in shortest_path:
    	shortest_path_id += [page_namesd.filter(lambda (K, V): K==i).collect]

    print shortest_path_id