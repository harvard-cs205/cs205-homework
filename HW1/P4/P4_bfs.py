from pyspark import SparkContext
from pyspark import AccumulatorParam

class setAccPar(AccumulatorParam):
    def zero(self, initialValue):
        return initialValue

    def addInPlace(self, s1, s2):
        s1 |= s2
        return s1


def rdd_bfs(sc, char_adjacent, root):
    touched = sc.accumulator( {root}, setAccPar() )
    this_level = {root}
    i = 0 # distance
    while(True):
        i += 1
        next_level = sc.accumulator( set(), setAccPar() )    
        next_level_rdd = char_adjacent.filter(lambda p: p[0] in this_level) #select the nodes in this level

        #next_level = set.union(* next_level_rdd.values().collect() ) #select the neighbors of the nodes in this level
        next_level_rdd.values().foreach(lambda x: next_level.add(x)) #select the neighbors of the nodes in this level
##        print next_level.value
##        print touched.value
##        print next_level.value - touched.value
        this_level = next_level.value - touched.value #remove visited nodes
        next_level_rdd.values().foreach(lambda x: touched.add(x)) #add in touched nodes
        #touched = touched | next_level #include the nodes in next level in touched set
        if len(this_level) == 0: #if all neighbors were visited, end this loop
            print "no new node!"
            print root, "diameter:", i
            break
    print root, "reachable characters:", len(touched.value)

