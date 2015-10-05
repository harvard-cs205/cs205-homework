from P4_bfs import * 
import pyspark
from pyspark import SparkContext
sc = SparkContext("local[4]")
sc.setLogLevel("ERROR")

#import data. split lines into comic books and superheroes..
data = sc.textFile('data.txt')
Superheroes = data.map(lambda word: word.split('",')[0]).map(lambda word: word.strip('"').encode("utf-8"))
Comics= data.map(lambda word: word.split('",')[1]).map(lambda word: word.strip('"').encode("utf-8"))

RDD1= Comics.zip(Superheroes)
RDD2 = RDD1.join(RDD1) #K,(V,W) = Comic, (Superhero,Superhero)

#create edge list and ensure no one is linked to themselves (this assumes
#every character has at least one neighbor otherwise if they are solo then
#they will never be found as they are omitted from this graph)...
Graph = RDD2.filter(lambda x: x[1][0]!=x[1][1]).map(lambda x: (x[1][0],x[1][1])).distinct().partitionBy(8).cache()

SOURCE = 'ORWELL'
SOURCE2 = 'MISS THING/MARY'
SOURCE3 = 'CAPTAIN AMERICA'

Orwell_count=bfs(SOURCE,Graph,sc)
Miss_thing_mary_count=bfs(SOURCE2,Graph,sc)
Captain_America_count=bfs(SOURCE3,Graph,sc)

print Orwell_count
print Miss_thing_mary_count
print Captain_America_count