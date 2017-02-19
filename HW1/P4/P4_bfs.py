def BFS(character, graph, accum):


    # Commented out the max diameter assumptions
    # diameter_max = 10
    # diameter = 0
    
    tocheck = [character]
    checked = []

    while len(tocheck)!=0: # and diameter<diameter_max:
        
        # get the neighbors of those in the tocheck list and put them into a neighbors list
        neighbors = graph.filter(lambda l: l[0] in tocheck).map(lambda l:l[1]).reduce(lambda a,b: a+[i for i in b if i not in a])

        # clean up and put it into checked list and rerun this for unsearched characters.
        unique_neighbors = [i for i in neighbors if (i not in checked and i not in tocheck and i!=character)]
        checked = checked + tocheck
        tocheck = unique_neighbors
        # diameter +=1
        accum.add(1)
        
    return [character, len(checked), checked]