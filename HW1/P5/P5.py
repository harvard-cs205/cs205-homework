import time
import pyspark
sc = pyspark.SparkContext()


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)
    
quiet_logs(sc)



''' PART 1: KEVIN BACON '''

def update_distance(d):
    ''' updates distance value for nodes when touched by BFS '''
    return lambda x: (x,d+1)

def check_match(t):
    return lambda x: x[0] == t


timeA=time.time() # benchmarking

n_parts = 40 # num partitions

links_fname  = "s3://Harvard-CS205/wikipedia/links-simple-sorted.txt" 
titles_fname = "s3://Harvard-CS205/wikipedia/titles-sorted.txt"

links = sc.textFile(links_fname)
titles = sc.textFile(titles_fname)

finder     = sc.accumulator(0) # start at one for single-source root
counter    = sc.accumulator(0) # start at one for single-source root
distance   =  0
source     = 2729536 # Kevin_Bacon
target     = 2152782 # Harvard_University

graph = (links.map(lambda x: x.split(" "))
              .map(lambda x: (int(x[0][:-1]),(x[1:],None)))
         )

root_list = (titles.zipWithIndex()
                   .map(lambda x: tuple(reversed(x)))
                   .filter(lambda x: x[0]==source)
                   .map( lambda x: (x[0], 0) )
             )

master_root_list = root_list

while finder.value == 0:
    print "Distance:",distance
    root_list = (graph.join( root_list )
                         .flatMap( lambda x: x[1][0][0], preservesPartitioning=True ) 
                         .map( update_distance(distance),  preservesPartitioning=True )
                         .leftOuterJoin( master_root_list, numPartitions=n_parts )
                         .filter( lambda x: x[1][1] is None )
                         .map( lambda x: (int(x[0]),x[1][0]),   preservesPartitioning=True )
                         .distinct(numPartitions=n_parts)
                    )
    master_root_list = master_root_list.union( root_list )
    distance += 1
    counter.add(1)
    root_list.filter(check_match(target)).foreach( lambda x: finder.add(1) )
    
timeB=time.time()
print "Total time:",timeB-timeA
print "found target in", str(counter.value),"degrees from root!"



''' PART 2: CONNECTED COMPONENTS '''


