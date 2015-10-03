import findspark
findspark.init()
import pyspark

# shut down the previous spark context
#sc.stop() 
sc = pyspark.SparkContext(appName="myAppName")
sc.setLogLevel('WARN')
import pdb
from P4_bfs import mybfs
import itertools as it

# Load data

wlist = sc.textFile('P4_data.txt',  use_unicode=True)

# Convert into (keys, values) with key= comic book  values = characters in the comic book
wlist = wlist.map(lambda x : x.split('"') )
comlist = wlist.map(lambda x : (x[3],x[1]) ).groupByKey()

# Create a rdd with keys = characters and values = neighbords of the key/character
newcom = comlist.values().map(lambda x: [char for char in x] )
newcom = newcom.map(lambda x: list(it.permutations(x,2)) ).map(lambda x: (1,list([i for i in x]))  ).flatMapValues(lambda x : x).values().groupByKey().sortByKey()
nodes = newcom.map(lambda v :  (v[0],[i for i in v[1]]   ) )

# Call the BFS algorithm for CAPTAIN AMERICA
nodesi, nodesf, distance = mybfs(nodes,'CAPTAIN AMERICA')
print 'SS-BFS for CAPTAIN AMERICA'
print 'distance: ', distance
print 'number of connected characters: ', nodesi-nodesf

# Call the BFS algorithm for MISS THING/MARY
nodesi, nodesf, distance = mybfs(nodes,'MISS THING/MARY')
print 'SS-BFS for MISS THING/MARY'
print 'distance: ', distance
print 'number of connected characters: ', nodesi-nodesf

# Call the BFS algorithm
nodesi, nodesf, distance = mybfs(nodes,'ORWELL' )
print 'SS-BFS for ORWELL'
print 'distance: ', distance
print 'number of connected characters: ', nodesi-nodesf