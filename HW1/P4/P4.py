from pyspark import SparkContext
from P4_bfs import *

def csvsplit(line):
    charac, comic = line[1:-1].split('","')
    return comic, charac

def removeselfedge(p):
    charkey = p[0]
    charset = p[1]
    charset.discard(charkey)
    return charkey, charset


if __name__ == '__main__':
    sc = SparkContext("local", appName="Spark1")
    sc.setLogLevel('WARN')
    
    csvpair = sc.textFile('source.csv', 32, use_unicode=False) #load csv
    cleanpair = csvpair.map(csvsplit) #clean to make (char, comic) pair
    char_edges = cleanpair.join(cleanpair).values() #join to make all edges (char1, char2)
    char_edges_set = char_edges.map(lambda p: (p[0],{p[1]})) #change value to set type
    char_adjacent_t = char_edges_set.reduceByKey(lambda v1, v2: v1|v2) #reduce to make a graph represented by adjacent lists
    char_adjacent = char_adjacent_t.map(removeselfedge).repartition(13).cache() #remove self edges and cache
    
    for root in ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']:
        rdd_bfs(sc, char_adjacent, root)
