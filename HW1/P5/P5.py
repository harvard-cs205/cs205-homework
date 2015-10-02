import pyspark
sc = pyspark.SparkContext()

import time

def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)
    
quiet_logs(sc)

def update_distance(d):
    ''' updates distance value for nodes when touched by BFS '''
    return lambda x: (x,d+1)

n_parts = 40 # num partitions

links_fname = "s3://Harvard-CS205/wikipedia/links-simple-sorted.txt"
titles_fname = "s3://Harvard-CS205/wikipedia/titles-sorted.txt"

titles = (sc.textFile(titles_fname,n_parts)
            .zipWithIndex()
            .map(lambda x: tuple(reversed(x)))
            .cache()
          )
graph  = (sc.textFile(links_fname,n_parts)
             .map(lambda x: x.split(" "))
             .map(lambda x: (int(x[0][:-1]),(x[1:],None)))
             .partitionBy(n_parts)
             .cache()
         )

finder     = sc.accumulator(0) # start at one for single-source root
counter    = sc.accumulator(0) # start at one for single-source root
distance   = 0

pairings      = {}
pairings["0"] = {"source":"Kevin_Bacon", "target":"Harvard_University"} # 2729536 --> 2152782
pairings["1"] = {"target":"Harvard_University", "source":"Kevin_Bacon"} # 2152782 --> 2729536

''' bonus pairings '''
'''
pairings["1"] = {"source":"Kevin_Bacon",    "target":"Captain_America"}        # 8908 --> 913762
pairings["2"] = {"source":"Captain_America","target":"O_Captain!_My_Captain!"} # 913762 --> 3702452
pairings["3"] = {"source":"Hadoop",         "target":"Data_(Star_Trek)"}       # 2111690 --> 1299323
'''

for pair in pairings.values():
    
    timeA = time.time() # benchmarking

    source = pair["source"]
    target = pair["target"]
    
    print "Find path:",source,"-->",target
    
    target_node = titles.filter(lambda x: x[1]==target).map( lambda x: (x[0], 0) ).partitionBy(n_parts).cache()
    
    root_list   = (titles.filter(lambda x: x[1]==source)
                         .map( lambda x: (x[0], 0) )
                         .partitionBy(n_parts)
                         .cache()
             )

    master_root_list = root_list.cache()
    master_chains    = root_list.map(lambda x: (x[0], None)).partitionBy(n_parts).cache()
    
    while finder.value == 0:
        graph2 = graph.join( root_list, numPartitions=n_parts ).cache()
        master_chains = (master_chains.union(graph2.flatMap(lambda x: [(int(node),x[0]) for node in x[1][0][0]]))
                                     .distinct()
                                     .partitionBy(n_parts)
                                     .cache()
                         )
        
        root_list = (graph2.flatMap( lambda x: x[1][0][0], preservesPartitioning=True )
                            .map( update_distance(distance),  preservesPartitioning=True )
                            .leftOuterJoin( master_root_list, numPartitions=n_parts )
                            .filter( lambda x: x[1][1] is None )
                            .map( lambda x: (int(x[0]),x[1][0]),   preservesPartitioning=True )
                            .distinct(numPartitions=n_parts)
                        )
        master_root_list = master_root_list.union( root_list )
        distance += 1
        counter.add(1)
        root_list.join(target_node).foreach( lambda x: finder.add(1) )
    
    countback = distance
    link_node = target_node
    path = target_node.map(lambda x: (x[0],distance))
    
    while countback > 0:
        link_node = master_chains.join(link_node, numPartitions=n_parts).map(lambda x: (x[1][0],0))
        path = path.union(link_node.map(lambda x: (x[0],countback-1)))
        countback -= 1


    timeB=time.time()        # benchmarking
    time_diff = timeB-timeA  # benchmarking

    print "Report: {} --> {}".format(source,target)
    print
    print "Found target in", str(counter.value),"degrees from root!"
    print "Full path:"
    print (titles.join(path)
                 .map(lambda x: (x[1][1],x[1][0]))
                 .sortByKey()
                 .values()
                 .collect()
           )
    print
    print "Total time:",time_diff

