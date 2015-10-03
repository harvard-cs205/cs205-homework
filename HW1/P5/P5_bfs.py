from pyspark import SparkContext
from pyspark import AccumulatorParam

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = set( int(x) for x in dests.split(' ') )
    return (int(src), dests)


class setAccPar(AccumulatorParam):
    def zero(self, initialValue):
        return initialValue

    def addInPlace(self, s1, s2):
        s1 |= s2
        return s1

def pointparent(p):
    children = p[1]
    parent = p[0]
    for child in children:
        yield ( child, parent )


if __name__ == '__main__':
    sc = SparkContext("local", appName="Spark1")
    sc.setLogLevel('WARN')
    
    #txtfile = sc.textFile('number.txt', 32) #test case
    txtfile = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)
    
    adjacent_graph = txtfile.map(link_string_to_KV).cache() #transform to graph
    nameindex = page_names.zipWithIndex().map(lambda p: (p[0], p[1]+1)) #build index table
    
    root = nameindex.lookup('Kevin_Bacon')[0]
    dest = nameindex.lookup('Harvard_University')[0]
    
    touched = set([root])
    this_level = set([root])
    parentrdd = sc.parallelize( [(root, -1)] ) #parent child pair, root has no parent
    i = 0 # distance
    while(True):
        next_level = sc.accumulator( set(), setAccPar() )
        next_level_rdd = adjacent_graph.filter(lambda p: p[0] in this_level) #select the nodes in this level
        next_level_rdd.values().foreach(lambda x: next_level.add(x)) #select the neighbors of the nodes in this level

        #child2parent = next_level_rdd.flatMap( pointparent ).filter( lambda p: p[0] not in touched ) #generate pair ( child, parent )
        child2parent = next_level_rdd.flatMap( pointparent ).reduceByKey(lambda v1, v2: v1 ) #generate pair ( child, parent )
        #print child2parent.collect()
        parentrdd = child2parent.subtractByKey(parentrdd).union(parentrdd) # preserve the parent of each node, set the parent of root to -1
        #parentrdd = child2parent.reduceByKey(lambda v1, v2: v1 ).union(parentrdd) #reserve only one parent and union back to parent rdd
##        print parentrdd.collect()
##        print next_level.value
##        print touched.value
##        print next_level.value - touched.value
##        print i,"==============================================================================================================================================================="
        this_level = next_level.value - touched #remove visited nodes
        touched |= next_level.value #add in touched nodes
        #touched = touched | next_level #include the nodes in next level in touched set
        if len(this_level) == 0: #if all neighbors were visited, end this loop
            print "not found!"
            print root, "diameter:", i
            break
        i += 1
        if dest in touched:
            print "reach destination!"
            print root, "diameter:", i
            break
        print "explored nodes in distance", i
        
    parentrdd = parentrdd.sortByKey()
    #print parentrdd.collect()
    
    # back trace the shortest path
    track = [dest]
    while (track[-1] != root):
        track.append( parentrdd.lookup(track[-1])[0] )
        
    indexname = nameindex.map( lambda p: (p[1], p[0]) ).sortByKey()

    for x in track[::-1]: #reverse list
        print indexname.lookup(x)[0]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        