import pyspark
import csv
from P4_bfs import *
import time

def _buildGraph(filename, sc):
    csvRdd = sc.textFile(filename)

    def comicCharMap(line):
        char, comic = line[1:-1].split('","')
        return (comic, char)

    # (comic, character)
    comicCharRdd = csvRdd.map(comicCharMap)
    # (charFrom, set(charTo)), where charFrom and charTo belongs to the same comic
    fromToRdd = comicCharRdd.join(comicCharRdd).map(lambda (_, (f, t)): (f, set([t])))
    # (char, adjacent character set)
    graph = fromToRdd.reduceByKey(lambda set1, set2: set1 | set2)

    def dropSelfMap(entry):
        char, adjacent = entry
        adjacent.discard(char)
        return (char, list(adjacent))

    graph = graph.map(dropSelfMap)
    return graph

if __name__ == '__main__':
    sc = pyspark.SparkContext(appName='YK-P4')

    filename = 'source.csv' 
    graph = _buildGraph(filename, sc)
    graph.cache()
    
    for name in ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']:
        touched, dist = bfs(graph, name)
        print '{0}: touched: {1}, diameter: {2}'.format(name, touched.count(), dist)
    #_writeDistDict(distDict, 'CAA') 

