import pyspark
sc = pyspark.SparkContext()
sc.setLogLevel("ERROR")

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

def shortest_path(page_data,result_rdd,dest_node,path_length):
    page_data = page_data.map(lambda x: (x[1],x[0]))
    path_val = [(page_data.lookup(dest_node)[0],path_length)]
    node = [dest_node]
    path = [dest_node]
    while path_length > 0:
        path_tuple = result_rdd.filter(lambda x: x[0][1] in node and x[1] == path_length).collect()
        node = []
        path_length -= 1
        for paths in path_tuple:
            node.append(paths[0][0])
        node = list(set(node))
        n_list = []
        for n in node: 
            n_list.append((page_data.lookup(n)[0],path_length))
        path.append(node)
        path_val.append(n_list)
    path.reverse()
    path_val.reverse()
    return path,path_val

def BFS_distance(rdd, source_node, dest_node):
    out_rdd = sc.parallelize([(source_node,0)])
    edge_rdd = sc.parallelize([])
    flag = True
    counter = 0
    while flag:
        counter += 1
        n_RDD = out_rdd.filter(lambda x:x[1] == counter-1).join(rdd)
        k_RDD = n_RDD.map(lambda x:(x[0],x[1][1])).flatMapValues(lambda x: [t for t in x])
        k_RDD = k_RDD.map(lambda x:(x,counter))
        edge_rdd = edge_rdd.union(k_RDD)
        n_RDD = n_RDD.flatMap(lambda x: x[1][1]).distinct()
        #print out_rdd.count()
        if n_RDD.isEmpty():
            break
        if n_RDD.filter(lambda x:x == dest_node).isEmpty() == False:
            flag = False
        n_RDD = n_RDD.map(lambda x:(x,counter))
        out_rdd = out_rdd.union(n_RDD)
        out_rdd = out_rdd.reduceByKey(lambda x,y:min(x,y))
    if flag == True:
        return "No solution found"
    else:
        return counter,out_rdd,edge_rdd
    return counter,out_rdd,edge_rdd


if __name__ == "__main__":
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
    #links = sc.textFile('testdata.txt')
    #page_names = sc.textFile('testpagedata.txt')
    neighbor_graph = links.map(link_string_to_KV)
    page_data = page_names.zipWithIndex().map(lambda x:(x[0],x[1]+1))

    #Finding Distance between Kevin Bacon and Harvard University
    source_node = page_data.lookup('Kevin_Bacon')[0]
    dest_node = page_data.lookup('Harvard_University')[0]
    print source_node
    print dest_node
    path_length,out_rdd,edge_rdd = BFS_distance(neighbor_graph,source_node,dest_node)
    path,path_val = shortest_path(page_data,edge_rdd,dest_node,path_length)
    print path_val
    print path
    print path_length
    
    print "Calculating reverse path"
    path_length,out_rdd,edge_rdd = BFS_distance(neighbor_graph,dest_node,source_node)
    path,path_val = shortest_path(page_data,edge_rdd,source_node,path_length)
    print path_val
    print path
    print path_length


