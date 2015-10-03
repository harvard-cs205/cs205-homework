# Initialize SC context
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark 1")
import numpy as np
from P4_bfs import *

# Your code here
if __name__ == '__main__':
  # Set up Initial RDDs
  MC_data = sc.textFile("data.txt",100)
  rdd_data = MC_data.map(lambda dataset: (dataset.split("\",\"")[0].split("\"")[1],dataset.split("\",\"")[1].split("\"")[0]))
  # rdd_graph will be (K, V) RDD with K = node and V = [connecting nodes]
  rdd_graph = rdd_data.groupBy(lambda data: data[1])\
            .map(lambda x: [cha for cha, comic in list(x[1])])\
            .flatMap(lambda cha_list: [(item,list(set(cha_list)-set([item]))) for item in cha_list])\
            .reduceByKey(lambda x,y: list(set(x+y)))

  # Run BFS
  # results will be (node, distance)
  result_CA = BFS(rdd_graph, "CAPTAIN AMERICA", sc)
  result_MT = BFS(rdd_graph, "MISS THING/MARY", sc)
  result_OR = BFS(rdd_graph, "ORWELL", sc)

