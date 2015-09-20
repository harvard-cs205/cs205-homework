import pyspark
import csv
from P4_bfs import *
import time

def _buildGraph(filename):
    issueNameDict = {}
    with open(filename, 'r') as fh:
        reader = csv.reader(fh, delimiter=',')
        for row in reader:
            name, issue = row
            if issue not in issueNameDict:
                issueNameDict[issue] = set()
            issueNameDict[issue].add(name)

    graph = {}
    for _, names in issueNameDict.iteritems():
        names = list(names)
        for i in xrange(len(names)):
            name = names[i]
            if name not in graph:
                graph[name] = []
            graph[name] += names[:i] + names[i+1:]

    for name, adjlist in graph.iteritems():
        graph[name] = list(set(adjlist))

    graph = [(k, v) for k, v in graph.iteritems()]
    return graph

def _writeDistDict(distDict, filename):
    with open(filename, 'w') as fh:
        for k, d in distDict.iteritems():
            fh.write('{0}: {1}\n'.format(k, d))

if __name__ == '__main__':
    sc = pyspark.SparkContext(appName='YK-P4')

    filename = 'source.csv' 
    graph = _buildGraph(filename)
    graph = sc.parallelize(graph)
    
    #name = 'CAPTAIN AMERICA'
    name = 'MISS THING/MARY'
    #name = 'ORWELL'
    result = bfs(graph, name, sc)
    # time.sleep(5)
    print result.count()
    #_writeDistDict(distDict, 'CAA') 

