def update_distance(d):
    ''' updates distance value for nodes when touched by BFS '''
    return lambda x: (x,d+1)

def ssbfs(sc,name_assoc,root,n_parts,doPrint=False):
    ''' Runs single-source breadth-first search on Marvel comic universe '''

    root_list        = name_assoc.filter( lambda x: x[0] == root ).map( lambda x: (x[0], 0) )   # get root list as RDD
    master_root_list = root_list.partitionBy(n_parts)                                           # master root list RDD
    
    counter    = sc.accumulator(1) # start at one for single-source root
    last_count = -1
    distance   =  0
    
    while counter.value > last_count:
        root_list = (name_assoc.join( root_list )                                       # get only current root node rows
                        .flatMap( lambda x: x[1][0],      preservesPartitioning=True )  # reduce to only assoc nodes
                        .map( update_distance(distance),  preservesPartitioning=True )  # update distance scores for each assoc node
                        .leftOuterJoin( master_root_list, numPartitions=n_parts )       # outer join with master_root 
                        .filter( lambda x: x[1][1] is None )                            # only keep nodes with no distance scores on master root list
                        .map( lambda x: (x[0],x[1][0]),   preservesPartitioning=True )  # bring back to normal root_list format (get rid of joined vals)
                        .distinct(numPartitions=n_parts)
                     )
        master_root_list = master_root_list.union( root_list ) # expand master root with union of current roots
        distance += 1
        last_count = counter.value
        root_list.foreach( lambda x: counter.add(1) )

    if doPrint:
        print "Report for SS-BFS with root node:", root
        print "Finished with total count: {}".format(counter.value)