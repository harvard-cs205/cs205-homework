# 1)We first create the graph
# 2)we also define a new BFS function taylored for connected components
# + a function that symmetrize the rdd for the 2nd part of the question
# 3)to finish  we show the script used and run for hours to get the given results

# 1)from https://github.com/thouis/SparkPageRank/blob/master/PageRank.py modified
# helper function to get a graph rdd
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

# process links into (node #, [neighbor node #, neighbor node #, ...]
neighbor_graph = links.map(link_string_to_KV)

# create an RDD for looking up page names from numbers
# remember that it's all 1-indexed
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()

#######################################################################
# set up partitioning - we have roughly 4 workers, if we're on AWS with 4
# nodes not counting the driver.  This is 8 partitions per worker.
# As we do an union afterwards we don't want it to be to big
# Cache this result, so we don't recompute the link_string_to_KV() each time.
#######################################################################

neighbor_graph = neighbor_graph.partitionBy(32).cache()

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


#2)
def distance_cc(sc, gc_final, root, db=None):
    '''
    This function does the breadth first search of a graph
    It filters out the visited nodes on the fly to avoid too long map reduce
    when the graphs get smaller
    Input:
    sc: spark context
    gc_final: rdd with the graph as described in the question
    root: first node that is visited (e.g "Captain America")
    db: within the connected components search, 
    we want to start with the graph that has less items
    Output:
    db: a RDD with the following structure
    (key=name of a hero,
    value =[distance to the root, [list of connexion], flag: 0 if not seen, 1 if marked, 2 if visited])
    with less items than the beginning one
    d-2 is the diameter
    nd: is the number of nodes in the graph
    end-begin: time it took to compute
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

    def selectRoot_db(k,v,root):
        """
        variation of selectRoot
        when db is already set
        Inputs:
        k: hero name
        v: list of his connection
        root: of the graph
        Outputs:
        instance of the new RDD
        if it is the root, it marks it and give it a distance zero to itself
        else it is non visited and it has an infinite distance to the root
        """
        if k!=root:
            return (k,v)
        else:
            return (k,[0,v[1],1])
        
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
    
    # initialize the diameter
    d=0
    # initialize the number of nodes
    nd = 0
    # initialize the accumulator
    stop = sc.accumulator(0)
    # initialize the RDD see helper functions
    if db is None:
        db = gc_final.map(lambda (k,v) :selectRoot(k,v,root))
    else:
        db = db.map(lambda (k,v) :selectRoot_db(k,v,root))
        
    while stop.value==0:
        # update diameter
        d+=1
        
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
        
        ## Taylored part for Connected Components
        ## NB: this will slow the code at the beginning when a lot of nodes are visited
        ## once at a time
        ## but it dramatically boosts the performance afterwards when there is less items less
        # get a list of the nodes that are visited
        pool=set(db.filter(lambda (k,v): v[2]==2).keys().collect())
        # get rid of them in the rdd
        db = db.filter(lambda (k,v): v[2]!=2).cache()
        # update the number of nodes in the graph
        nd+=len(pool)
        # suppress the suppressed nodes from the list of connexions
        # so that they don't reappear
        db = db.map(lambda (k,v):(k,[v[0],list(set(v[1]).difference(pool)),v[2]]))
        
    end = time.time()
    return db, d-2, nd, end-begin

################
# Helper functions and function to symmetrize the RDD
################
def duplicate_flat(k,v):
    """
    Create 2 pairs from a key and one connexion
    """
    output = []
    for b in v:
        output.append((k,b))
        output.append((b,k))
    return output

def keep_duplicate(v):
    """
    When aggregated if there is no duplicate 
    it means that the link is not symmetric
    """
    output = []
    v.sort()
    for k in range(len(v)-1):
        if v[k]==v[k+1]:
            output.append(v[k])
    return output

def keep_symmetric(sc, gc_final):
    """Returns the needed symmetrized  RDD"""
    maybesymmetric = gc_final.flatMap(lambda (k,v): duplicate_flat(k,v))
    to_check = maybesymmetric.groupByKey().mapValues(list)
    gc_symmetric = to_check.map(lambda (k,v): (k,keep_duplicate(v)))
    return gc_symmetric


# 3)
# Nota Bene: this code is not within a function 
# as I interupted it to get some insights on the results
# I add to update the variables within the console on the fly

## Those lines are normally function arguments
#initialize the root
root = Kevin_Bacon
#initialize gc_final
gc_final=neighbor_graph

# initialize the number of connected components
num_cc = 0
# initialize the number of nodes in the biggest graph
max_nbnode = 0
# use a python variable variable
# NB: we could have used an accumulator as in P4_bfs.py
stop = False

# first search in the graph which root is given
num_cc+=1
# distance_cc.py function is explained above
# of most importance all the visited nodes are filtered out from db 
# on the fly within the function, it fastens the unions
db, d, nd,t = distance_cc(sc, gc_final, root)
# update max_nbnode
if nd>max_nbnode: 
    max_nbnode=nd
# stop if there is no more node to visit    
if db.take(1)==[]:
    stop = True
else:
    root = db.take(1)[0][0]


while not stop:
    # increment the number of connected components
    num_cc+=1
    # visit a new graph
    db, d, nd,t = distance_cc(sc, gc_final, root, db)
    # update max_nbnode
    if nd>max_nbnode:
        max_nbnode=nd
    # if there is no more nodes we are done visiting
    if db.take(1)==[]:
        stop = True
    else:
        root = db.take(1)[0][0]
