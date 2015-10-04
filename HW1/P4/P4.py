from P4_bfs import *
from pyspark import SparkContext
from urllib2 import urlopen
import os

if __name__=='__main__':
    NPART = 4*16
    characters = [u'CAPTAIN AMERICA', u'ORWELL', u'MISS THING/MARY']
    # initialize spark
    sc = SparkContext()

    # save text file
    datafile = 'source.csv'
    if not os.path.exists(datafile):
        # download text file
        url = 'http://exposedata.com/marvel/data/source.csv'
        with open(datafile, 'wb') as f:
            f.write(urlopen(url).read())

    # load data into a rdd (issue, character)
    data = sc.textFile(datafile, 32).map(
        lambda line: tuple([w.strip('"') for w in 
            line.split('","')])[::-1])

    # self join to create edges
    edges = data.join(data)

    # remove duplicates, i.e. character with itself
    edges = edges.values()
    edges = edges.filter(lambda (chr1, chr2): chr1 != chr2)

    # create graph rdd (character, {all linked characters})
    graph = edges.groupByKey().mapValues(set).repartition(NPART).cache()

    # serial implemention
    # print 'Serial implemention'
    # for root in characters:
    #     num_traversed = bfs_serial(graph, root)
    #     print root
    #     print 'Nodes touched:', num_traversed

    # parallel implementation
    print 'Parallel implemention'
    for root in characters:
        bfs_parallel(sc, graph, root)
        num_traversed = bfs_parallel(sc, graph, root)
        print root
        print 'Nodes touched:', num_traversed
        print


