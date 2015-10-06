from pyspark import AccumulatorParam

class SetAccumulator(AccumulatorParam):
    def zero(self, initialValue):
        return set()

    # union sets for visited
    def addInPlace(self, s1, s2):
        s1 |= s2
        return s1

# find the number of reachable nodes from starting character
def ss_bfs(sc, rdd, character):
	# initialize variables
    visited = sc.accumulator(set(), SetAccumulator())
    frontier = {character}
    distance = 0
    
    # terminate if found nothing new
    while len(frontier):
    	# update values
        distance += 1
        prev_vals = set(list(visited.value))

        # iterate through current level and update visited 
        rdd.filter(lambda (x,y): x in frontier)
           .foreach(lambda (x,y): visited.add(y))
        
        # update frontier
        frontier = visited.value - prev_vals

    return len(visited.value)