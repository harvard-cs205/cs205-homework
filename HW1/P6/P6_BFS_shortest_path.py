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
            print tocheck
            for i in tocheck:
                try:
                    neighbors = graph.filter(lambda (k,v): k==i).collect()[0][1]
                    # clean up neighbor for only unique, never visited 
                    unique_neighbors = []
                    for j in neighbors:
                        if j not in checked:# This means j is a unique neighbor never visited
                            if j==end:
                                found = True#found!
                            unique_neighbors += [j]
                            checked += [j]

                except:# if no key, will raise IndexError
                    unique_neighbors = []

                #print unique_neighbors
                unique_tree[level][i] = unique_neighbors
                all_neighbors += unique_neighbors
            
        tocheck = all_neighbors
        print level
        level +=1
        
    # Following code is tracing the path leading to the specific node given a unique graph.
    print unique_tree

    revpath = []
    searchfor = end
    for i in range(len(unique_tree)):
        levelindex = len(unique_tree) - 1 - i
        print unique_tree[levelindex]
        for key in unique_tree[levelindex]:
            if  searchfor in unique_tree[levelindex][key]:
                revpath += [key]
                searchfor = key
    path = list(reversed(revpath))
    
    print path
    return path