from pyspark import SparkContext
from P4_bfs import * 
import time

# given a list, find all pairwise tuples
#NOTE: this will naturally filter out comic books with only 1 superhero in it
#This is not a problem as there is no relationships involved
def split_list_to_tuples(x):
	newList = []
	for a in x:
		for b in x:
			if a != b:
				newList.append((a,b))
	return newList


#initialize spark
sc = SparkContext('local', "Anagram")

#get the raw data
raw_data = "/home/zhiqian/Dropbox/CS205/Homework/HW2_data/source.csv"
superheroes = sc.textFile(raw_data)

#Note: i split the string via ," as there are commas in the names of some superheroes
#eg. frost, carmilla "",""
superheroes2 = superheroes.map(lambda x: x.split(",\""))

#reverse the KV pair so that the comic issue becomes the key. Remove "" from name
comic_issue = superheroes2.map (lambda (x,y): (y,x[1:len(x)-1]))

#groupbyKey to condense all the superheroes in each comic book issue. 
comic_issue2 = comic_issue.groupByKey()


#split the list into pair-wise tuples. these are the vertices of the graph
vertices = comic_issue2.flatMap(lambda (x,y): split_list_to_tuples(y))

#remove duplicates
vertices2 = vertices.distinct()

vertices3 = vertices2.groupByKey()



#--------------------Sanity Checks. Please ignore------------------------
#print vertices3.count()
#print superheroes2.count()
#print comic_issue2.count()
#print vertices.count()
#print vertices2.mapValues(list).take(10)
#print vertices2.count()
#print vertices3.count()
#print vertices3.mapValues(list).take(2)
#--------------------------------------------------------------------------

start = time.time()
#Toggle for the different superheroes to start with
#results = bfs(vertices3, "ORWELL", sc)
#results = bfs(vertices3, "MISS THING/MARY", sc)
results = bfs(vertices3, "CAPTAIN AMERICA", sc)
print time.time()-start

print results
