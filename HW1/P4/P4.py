import findspark
findspark.init('/home/shenjeffrey/spark/')
import pyspark
import matplotlib.pyplot as plt
import seaborn as sns
from P4_bfs import shortest_path_parallel

# initiate spark
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

# Read in data
data = sc.textFile("source.csv")
data.take(10)

# Function to split the csv file
def clean_csv(line):
    line = line.split('","')
    line[0] = line[0].replace('"', '')
    line[1] = line[1].replace('"', '')
    return line

clean_csv('"FROST, CARMILLA","AA2 35"')

# Clean data 
# (Comic Book, Comic Book Character)
clean_data = data.map(clean_csv).map(lambda x: (x[1], x[0]))
clean_data.take(10)

# Create symmetric edges for a bidirectional graph
# 1) Inner join on Comic Book (this guarantees A-A, A-B, B-A, B-B edges)
# 2) Delete all duplicated pairs (delete A-A and B-B to get A-B and B-A edges)
edges = clean_data.join(clean_data)
edges = edges.filter(lambda(book, char): char[0] != char[1])
edges = edges.map(lambda(book, char_edge): char_edge)
edges.take(10)

# Create graph from edges
# GroupbyKey with Comic Book Character
graph = edges.groupByKey().mapValues(list)
graph = graph.repartition(16).cache()

# Final results
#orwell = shortest_path(graph, "ORWELL", 10)
#miss_thing = shortest_path(graph, "MISS THING/MARY", 10)
#captain_america = shortest_path(graph, "CAPTAIN AMERICA", 10)

print shortest_path_parallel(sc, graph, "ORWELL")
print shortest_path_parallel(sc, graph, "MISS THING/MARY")
print shortest_path_parallel(sc, graph, "CAPTAIN AMERICA")

