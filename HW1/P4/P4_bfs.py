import pyspark
from pyspark import SparkContext

def bfs(graph,character,sc):
    print 'You are searching for: ', character

    final_rdd1 = node1 = sc.parallelize([(character,0)])
    accumulator = sc.accumulator(0)

    while accumulator.value == 0:        
    
        node1 = graph.join(node1).distinct().values().mapValues(lambda x: x+1).subtractByKey(final_rdd1)
        final_rdd1 = final_rdd1.union(node1).cache() 

        if node1.isEmpty():
            accumulator.add(1)
            continue
    res = final_rdd1.groupByKey().mapValues(list)
     
    return res, res.count()
