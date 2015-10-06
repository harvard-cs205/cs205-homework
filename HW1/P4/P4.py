import findspark
findspark.init()
import pyspark
import csv

from P4_bfs import bfs


def construct_edges(x):
    # add each edge to list  
    i, s_list = x
    l = len(s_list)
    edges = []
    for i in range(l):
        for j in range(l):
            if i != j: 
                edges.append(((s_list[i], s_list[j]), None))
    return edges

def main():
    # queries are CAPTAIN AMERICA 
    # MISS THING/MARY
    # ORWELL
    sc = pyspark.SparkContext()
    sc.setLogLevel('WARN')
    source_reader = csv.reader(open("source.csv", 'rb'), delimiter=',')

    # readuce by key to get RDD with issue -> [super_hero list]
    issue_sh = sc.parallelize(list(source_reader), 100).map(lambda x: (x[1].strip(), [x[0].strip()])).reduceByKey(lambda x, y: x + y)

    edges = issue_sh.flatMap(construct_edges).distinct().map(lambda x: (x[0][0], [x[0][1]]))
    adj_list = edges.reduceByKey(lambda x, y: x + y).cache()
    
    print "ORWELL"
    print bfs(adj_list, "ORWELL", sc)
    print 

    print "CAPTAIN AMERICA"
    print bfs(adj_list, "CAPTAIN AMERICA", sc)
    print  

    print "MISS THING/MARY"
    print bfs(adj_list, "MISS THING/MARY", sc)
    print 

if __name__=="__main__":
    main()
