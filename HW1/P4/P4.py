from P4bfs import *
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="P4")


if __name__ == "__main__":
    char_list = sc.textFile("source.csv").map(lambda line: line.split('","'))\
                            .filter(lambda line: len(line)>1)\
                            .map(lambda line: (line[0],line[1]))\
                            .map(lambda x:(x[0][1:],x[1][:-1]))
    
    issue_list = char_list.map(lambda x:(x[1],[x[0]])).reduceByKey(lambda x,y:x+y)
    graph_list = char_list.map(lambda x:(x[1],x[0])).join(issue_list)
    graph_list = graph_list.map(lambda x:(x[1][0],x[1][1]))\
                            .reduceByKey(lambda x,y:x+y)\
                            .map(lambda x:(x[0],list(set(x[1]))))

    search_nodes = ['CAPTAIN AMERICA','MISS THING/MARY','ORWELL']
    output_list =[]

    for node in search_nodes:
        print node
        res = BFS_acc(sc,graph_list,node)
        output_list.append(res.count())

    print output_list
    


