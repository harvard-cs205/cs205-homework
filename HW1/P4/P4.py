from P4_bfs import *

import itertools

# Initialize Spark
from pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

# Helper function used to clean data loaded from CSV file
# Splits each line by comma, with error-trapping for names with commas
# Removes "" around names and strips out whitespace
def split_csv(val):
    split = val.split(',')
    name = ','.join(split[:-1]).replace('"','').strip()
    comic = split[-1].replace('"','')
    return tuple([comic, name])

# Helper function calculates all 2-character permutations for a comic book
def perm_chars(val):
    return list(itertools.permutations(val[1], 2))

# Load data
clist = sc.textFile('source.csv')

# Split data into (comic book, character) pairs
clist_split = clist.map(lambda x: split_csv(x)).distinct()

# Group by comic book: (comic book, (char #1, char #2, ...))
comic_allchars = clist_split.groupByKey().mapValues(list)

# Group by character: (character, (char #1, char #2, ...))
char_allchars = comic_allchars.flatMap(lambda x: perm_chars(x)).distinct().groupByKey().mapValues(list)

# Results of BFSS search for 3 given characters

char = 'CAPTAIN AMERICA'
nodes = bfs(sc, char_allchars, char)
print '%d touched nodes when starting from %s' %(nodes, char)

char = 'MISS THING/MARY'
nodes = bfs(sc, char_allchars, char)
print '%d touched nodes when starting from %s' %(nodes, char)

char = 'ORWELL'
nodes = bfs(sc, char_allchars, char)
print '%d touched nodes when starting from %s' %(nodes, char)