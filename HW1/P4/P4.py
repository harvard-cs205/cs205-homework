from pyspark import SparkContext
sc = SparkContext("local[8]")
import numpy as np
import csv
from P4_bfs import *

def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)

quiet_logs(sc)

def combine_characters(x):
    """
    Helper function to generate a list of (char1, char2) pairs where (char1, char2) is a pair <=> 
    char1 and char2 appear in an issue together. Also, (char1, char2) is a pair <=> (char2, char1) is a pair.
    """
    char_pairs = []
    for i in x:
        for j in x:
            if (i != j) and ((i,j) not in char_pairs): # We only check (i,j) because (i,j) in list <=> (j,i) in list
                char_pairs.append((i,j))
                char_pairs.append((j,i))
    return char_pairs

# Initialize everything
issue_list = []

with open('source.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        issue_list.append((row[1], [row[0]]))
        
search_list = ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']
        
char_pairs = sc.parallelize(issue_list).reduceByKey(lambda x,y:  x+y).flatMapValues(
    combine_characters).values().distinct()
char_pairs = char_pairs.partitionBy(20)
char_pairs = char_pairs.sortByKey().cache()

for start_char in search_list:
    print start_char, ':', bfs(char_pairs, start_char, sc).count()