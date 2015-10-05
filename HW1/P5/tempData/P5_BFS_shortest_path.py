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
    #print unique_tree

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
    
    print path
    
    return path +[end]