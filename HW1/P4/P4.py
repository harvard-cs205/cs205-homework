# Talked with Isadora Nun about strategy for this problem

import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from P4_bfs import ssbfs
sc = SparkContext()

# Build an rdd of issues with all their associated characters from the source.csv text file
issue_characters = sc.textFile('source.csv').map(lambda line: (line[line.rindex(',') + 1:].replace('"', ''), line[:line.rindex(',')].replace('"', '')))

def getOthers(char_name, names):
    others = []
    for name in names:
        if (name != char_name) & (name not in others):
            others.append(name)
    return others

# Build and rdd of characters ot all of the charaters they are related to
final = issue_characters.join(issue_characters).values().groupByKey().map(lambda chars: (chars[0], getOthers(chars[0], list(chars[1])))).partitionBy(8).cache()

# Run the searches
ssbfs('CAPTAIN AMERICA', final, sc)
ssbfs('MISS THING/MARY', final, sc)
ssbfs('ORWELL', final, sc)