import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="")
import random

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

## Parse the text files 
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

neighbor_graph = links.map(link_string_to_KV)
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()
neighbor_graph = neighbor_graph.partitionBy(256).cache()
Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
assert len(Kevin_Bacon) == 1
Kevin_Bacon = Kevin_Bacon[0][0]
Harvard_University = page_names.filter(lambda (K, V): V == 'Harvard_University').collect()
assert len(Harvard_University) == 1
Harvard_University = Harvard_University[0][0]
wiki_graph = neighbor_graph.map(lambda x: (x[0], (1000, x[1], []))).cache()

def bfs(sourceRDD, root, end, lookupRDD):
    ## Set the root
    def set_root((node, (dist, neighbors, prev))):
        if (node == root):
            return (node, (0, neighbors, prev))
        else:
            return ((node, (dist, neighbors, prev)))

    ## Update the node RDD: character, (distance, [list of neighbors], [list of previous nodes])
    def update_node(inputRDD, step, visit_count):
        ## First we find the current node's neighbors and create a row: neighbor, (current_distance + 1, [empty list])
        def find_neighbor((node, (dist, neighbors, prev))):
            next_list = [(node, (dist, neighbors, prev))]
            if dist == step:
                for neighbor in neighbors:
                    next_list.append((neighbor, (step+1, [], [node])))
            return next_list
        ## Then we combine the newly created neighbor with the old neighbor with the updated distance
        ## Note that the "prev" list will only contain the earliest reached nodes, which ensures that all nodes we store in [prev] belong to the shortest paths
        def update_neighbor((dist1, neighbors1, prev1), (dist2, neighbors2, prev2)):
            dist3 = min(dist1, dist2)
            neighbors3 = neighbors1 + neighbors2
            prev3 = []
            if dist1 == dist3:
                prev3 = prev3 + prev1
            if dist2 == dist3:
                prev3 = prev3 + prev2
            return (dist3, neighbors3, prev3)
        outputRDD = inputRDD.flatMap(find_neighbor).reduceByKey(update_neighbor)
        return outputRDD    

    ## Find the path between the start and end nodes, by searching backwards in the [list of previous nodes]
    def find_path(end, total_step, nodeRDD):
        ## Set the end node
        def set_end((node, (path, prev))):
                if (node == end):
                    return ((node, (total_step, prev)))
                else:
                    return ((node, (path, prev)))
        ## First we select a random node from "prev" list
        def update_path(inputRDD1, step):
            def find_prev((node, (path, prev))):
                prev_list = [(node, (path, prev))]
                if path == step:
                    pre = random.choice(prev)
                    prev_list.append((pre, (step-1, [])))
                return prev_list
            ## Then we merge it to the current RDD with the updated "path" variable - indicating at which step the node is reached
            outputRDD1 = inputRDD1.flatMap(find_prev).reduceByKey(lambda (x1,y1), (x2,y2): (max(x1,x2), y1+y2))
            return outputRDD1

        pathRDD = nodeRDD.map(lambda (node, (dist, neighbors, prev)): (node, (-1, prev)))
        pathRDD2 = pathRDD.map(set_end)

        #Loop through the steps by picking random previous nodes along the way, and output the picked path
        for i in range(total_step, 0, -1):
            pathRDD2 = update_path(pathRDD2, i)
#         path_found = pathRDD2.map(lambda (k, (v1, v2)): (v1, k)).filter(lambda (k, v): k >= 0).sortByKey()
        path_node = pathRDD2.filter(lambda (k, (v1, v2)): v1 >= 0).map(lambda (k, (v1, v2)): (v1, k)).sortByKey()
        return path_node    
    
    ## Initialize root and count variables
    nodeRDD = sourceRDD.map((set_root)).cache()
    new_count = 1
    visit_count = sc.accumulator(1)
    print "Start from root:", root
                                     
    step = 0
    while new_count > 0:
        nodeRDD = update_node(nodeRDD, step, visit_count).cache()
        new_count = nodeRDD.filter(lambda (k, v): v[0] == step+1).count()
        visit_count += new_count
        print "Step:", (step+1),"; New Nodes:", new_count,"; Total Nodes:", visit_count, "."
        reach_end = nodeRDD.filter(lambda (k, v): k == end and len(v[2]) > 0).cache()
        if reach_end.count() > 0:
            print "Reach the end:", end
            total_step = step+1
            path_node = find_path(end, total_step, nodeRDD)
            path_name_list = []
            for i in path_node.collect():
                path_name_list.append(lookupRDD.lookup(i[1]))
            path_name = [x[0] for x in path_name_list]
            print "The following path is found among all possible paths:"
            print path_node.collect()
            print path_name
            return (path_node, path_name)
        step += 1
        
result1 = bfs(wiki_graph, Harvard_University, Kevin_Bacon, page_names)
result2 = bfs(wiki_graph, Kevin_Bacon, Harvard_University, page_names)