import findspark
findspark.init('/home/toby/spark')
import pyspark
sc = pyspark.SparkContext(appName="Spark 2")
sc.setLogLevel('WARN') 

from P4_bfs import *
if __name__ == '__main__':

    myfile = "source.csv"
    graph = sc.textFile(myfile)

    def parser(string):
        words = string.split("\"")
        #return words[3], words[1]
        return words[3].strip(), words[1].strip()

    graph = graph.map(parser)
    graph = graph.join(graph) # pair up two characters
    graph = graph.filter(lambda (book, chars): chars[0]!=chars[1]) # delete self edge

    graph = graph.map(lambda (book, chars): (chars[0], set([chars[1]])))
    graph = graph.reduceByKey(lambda a, b: a|b) # [key, set(children)]
    graph = graph.partitionBy(16)


    characters = ["CAPTAIN AMERICA", "MISS THING/MARY", "ORWELL"]
    num_nodes = []
    for character in characters:
        num_nodes.append(BFS(character, graph, sc))
    for i, character in enumerate(characters):
        print character, "is connected by", num_nodes[i]
