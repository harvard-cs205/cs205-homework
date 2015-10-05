import pyspark

def BFS_shortest_path(graph, start, end):
    max_iteration = 10
    
    checked = []
    tocheck = [start]
    
    unique_tree = []
    level = 0
    
    found = False
    
    # find the unique tree that contains a path from start to end
    while not found and level<max_iteration:
        unique_tree += [dict()]
        if len(tocheck)!=0:
            all_neighbors = []
            print 'tocheck size: ', len(tocheck)
            neighbors_list = graph.filter(lambda (k,v): k in tocheck).collect()
            for i in neighbors_list:
                name = i[0]
                neighbors = i[1]
                print 'neighbor size: ', len(neighbors)
                # clean up neighbor for only unique, never visited 
                unique_neighbors = []
                for j in neighbors:
                    if j not in checked:# This means j is a unique neighbor never visited
                        unique_neighbors += [j]
                        checked += [j]
                        if j==end:
                            found = True#found!
                            break

                #print unique_neighbors
                unique_tree[level][name] = unique_neighbors
                all_neighbors += unique_neighbors
                if found==True:
                    break
            
        tocheck = all_neighbors
        print level
        level +=1
        
        if found==True:
            break
        
    # Following code is tracing the path leading to the specific node given a unique graph.
    #return unique_tree
    print unique_tree

    revpath = []
    searchfor = end
    for i in range(len(unique_tree)):
        levelindex = len(unique_tree) - 1 - i
        #print unique_tree[levelindex]
        for key in unique_tree[levelindex]:
            if  searchfor in unique_tree[levelindex][key]:
                revpath += [key]
                searchfor = key
    path = list(reversed(revpath))
    
    
    return path+[end]




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
        shortest_path_id += [page_names.filter(lambda (K, V): K==i).collect()]

    print shortest_path_id