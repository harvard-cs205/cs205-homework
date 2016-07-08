def distance_between(sc, gc_final, root, arrive):
    '''
    This function gives the distance between the arrive and the root node. 
    It stops when it arrives at the arrive node.
    '''
    import time #needed to compute the time
    import sys #neede to get the maximum integer of the machine
    
    begin = time.time()
    
    # initialization of the accumulator using a spark context
    stop = sc.accumulator(0)
    
    # Helper function for the BFS
    def selectRoot(k,v,root):
        """
        This function creates the new RDD as described above
        Inputs:
        k: hero name
        v: list of his connection
        root: of the graph
        Outputs:
        instance of the new RDD
        if it is the root, it marks it and give it a distance zero to itself
        else it is non visited and it has an infinite distance to the root
        """
        # flag=0 not visited, 1 marked, 2 visited
        if k!=root:
            # unvisited node have an infinite distance to the root, here it is maxint
            return (k,[sys.maxint,v,0])
        else:
            # root
            return (k,[0,v,1])
        
    def concatenateList(k,v):
        """
        This function updates the list 
        marks visited (flag 2) the visited nodes 
        and mak marked (flag 1) its connexion
        It returns a list of (k,v) to flat map
        Input:
        k: the visited node
        v: its connexion
        Output:
        list to flatMap
        """
        # update connexions as we don't know their connexion at this point we assign None
        list1 = [(b,[v[0]+1,None,1]) for b in v[1]]
        # update visited node
        list1.append((k,[v[0],v[1],2]))
        return list1
    
    def reduce_toMark(v1,v2):
        """
        transitive reduce function:
        Input:
        v1: first value assigned to the key
        v2: second value assigned to the same key
        Output:
        distance:is the minimum of the distance since we want the shortest path
        connexion: take the connexion from the database
        flag: is the maximum of the flag, if it is already visited it stays visited
        """
        # min of distance
        a1 = min(v1[0],v2[0])
        # max of flags
        a3 = max(v1[2],v2[2])
        # if the node is newly marked, its connexion is None 
        # so we want to assigne its actual connexion
        if v1[1] is None:
            a2 = v2[1]
        else:
            a2 = v1[1]
        return [a1,a2,a3]
    
    # update the RDD in the new format
    db = gc_final.map(lambda (k,v) :selectRoot(k,v,root))

    while stop.value==0:
        # select the marked items
        tomark = db.filter(lambda (k,v): v[2]==1)
        # if there is not any, change the accumulator to stop the loop
        if tomark.take(1)==[]:
            stop.add(1)
        # mark the connection of the currently visited nodes and mark those latter visited
        mark = tomark.flatMap(lambda (k,v): concatenateList(k,v))
        # union of the 2 RDD
        db = db.union(mark)
        # reduce as explain in the helper function and partition/cache for efficiency
        db = db.reduceByKey(reduce_toMark,numPartitions=23).cache()
        
        #DIFFERENCE WITH PBM4
        # find the value related to the arrival node
        arrived = db.lookup(arrive)
        # check if it has been visited
        if arrived[0][2]==2:
            # if yes, directly return its distance to the root node
            return arrived[0][0]
        
    end = time.time()
    #return the distance list
    return db, end-begin
