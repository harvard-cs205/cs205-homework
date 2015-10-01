from pyspark import SparkContext

def rdd_bfs(sc, char_adjacent, root):
    touched = {root}
    this_level = touched.copy()
    next_level = {}
    for i in xrange(10):
        next_level_rdd = char_adjacent.filter(lambda p: p[0] in this_level) #select the nodes in this level
        next_level = set.union(* next_level_rdd.values().collect() ) #select the neighbors of the nodes in this level
        next_level = next_level - touched #remove visited nodes
        touched = touched | next_level #include the nodes in next level in touched set
##            print i,"===================================="
##            print this_level
##            print next_level
        if len(next_level) == 0: #if all neighbors were visited, end this loop
            print "no new node!"
            print root, "diameter:", i
            break
        this_level = next_level.copy()
        next_level.clear()
    print root, "reachable characters:", len(touched)

##    a = char_adjacent.collect()
##    for i in xrange(5):
##        print a[i]