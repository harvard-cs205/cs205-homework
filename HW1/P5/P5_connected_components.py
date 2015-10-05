import pyspark
sc = pyspark.SparkContext()
sc.setLogLevel("ERROR")

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

def find_all_connected_graphs(rdd):
    def BFS(rdd, node):
        out_rdd = sc.parallelize([(node,1)])
        i = 1
        while True:
            n_RDD = out_rdd.filter(lambda x:x[1] == i).join(rdd)\
                    .flatMap(lambda x: x[1][1]).distinct()
            if n_RDD.isEmpty():
                break
            n_RDD = n_RDD.map(lambda x:(x,i+1))
            out_rdd = out_rdd.union(n_RDD)
            out_rdd = out_rdd.reduceByKey(lambda x,y:min(x,y))
            i += 1
        return out_rdd
    base_rdd = rdd.map(lambda x:(x[0],0))
    out_list = []
    while base_rdd.filter(lambda x:x[1] == 0).isEmpty() == False:
        start_node = base_rdd.filter(lambda x:x[1] == 0).take(1)[0]
        start_node = start_node[0]
        out_rdd = BFS(rdd,start_node)
        base_rdd = base_rdd.subtractByKey(out_rdd)
        out_list.append(out_rdd.count())
    return out_list

if __name__ == "__main__":
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt',32)
    #links = sc.textFile('testdata.txt')
    #page_names = sc.textFile('testpagedata.txt')
    neighbor_graph = links.map(link_string_to_KV)
    edge_graph = neighbor_graph.flatMapValues(lambda x:[t for t in x])

    rev_edge_graph = edge_graph.map(lambda x:(x[1],x[0]))
    symm_edge_graph = edge_graph.union(rev_edge_graph).distinct()
    symm_edge_graph = symm_edge_graph.map(lambda x:(x[0],[x[1]])).reduceByKey(lambda x,y:x+y)

    bidir_edge_graph = edge_graph.intersection(rev_edge_graph)
    bidir_edge_graph = bidir_edge_graph.map(lambda x:(x[0],[x[1]])).reduceByKey(lambda x,y:x+y)
    
    page_data = page_names.zipWithIndex().map(lambda x:(x[0],x[1]+1))
    bidir_graph_list = find_all_connected_graphs(bidir_edge_graph)
    print "Bidirectional"
    print bidir_graph_list
    print max(bidir_graph_list)

    symm_graph_list = find_all_connected_graphs(symm_edge_graph)
    print "Symmetric"
    print symm_graph_list
    print max(symm_graph_list)




