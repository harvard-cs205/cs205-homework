# References:
# http://hadooptutorial.wikispaces.com/Iterative+MapReduce+and+Counters
# http://www.slideshare.net/jhammerb/lec5-pagerank
# The CSV and Spark manual pages

import csv
import time
from pyspark import SparkContext
from P4_bfs import *

sc = SparkContext("local", "P4", pyFiles=['P4_bfs.py'])

issues = {}
charLookup = {}
revLookup = []
adjList = {}
heroCount = 0

# Read CSV file
with open('source.csv','rb') as f:
    reader = csv.reader(f)
    for row in reader:
        name, issue = row[0].strip(), row[1].strip()
        if issue not in issues:
            issues[issue] = set()
        if name not in charLookup:
            charLookup[name] = heroCount
            revLookup.append(name)
            heroCount += 1
        issues[issue].add(name)

# Create adjacency list
for cs in issues.values():
    chars = [charLookup[c] for c in cs]
    for i in xrange(len(chars)):
        c1 = chars[i]
        if c1 not in adjList:
            adjList[c1] = set()
        for j in xrange(i+1, len(chars)):
            c2 = chars[j]
            if c2 not in adjList:
                adjList[c2] = set()
            adjList[c1].add(c2)
            adjList[c2].add(c1)

sT = time.time()
sList = ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']
for char in sList:
    print 'Source =', char
    print 'Touched', nTouched(sc, adjList, charLookup[char]),'nodes'
print time.time() - sT
