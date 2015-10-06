import findspark
findspark.init()
import pyspark
import re
import csv

from P4_bfs import *

sc = pyspark.SparkContext(appName="P4")
sc.setLogLevel('ERROR')

#load in the list of characters and appearances in comic books
with open("source.csv", "rb") as inputfile:
	reader = csv.reader(inputfile, quotechar='"')
	character_book = []
	for line in reader:
		character_book.append(line)
	character_book = sc.parallelize(character_book)

#reverse the list as an association from book to character
book_character = character_book.map(lambda (x,y): (y,x))

#group by appearance in each comic book
by_book = book_character.groupByKey()

#for each comic book, create an edge from each character to each other character
def characterToOtherCharacters(current_book):
	book, characters = current_book
	characters = list(characters)

	edges = []
	
	for i in range(len(characters)):
		for j in range(len(characters)):
			if i!=j:
				edges.append((characters[i], characters[j]))
				edges.append((characters[j], characters[i]))

	return edges

#expand the list of characters by books to a flattened list of all edges
character_pairings = by_book.flatMap(characterToOtherCharacters)

#group by the first character to create an adjacency list
adjacency_list = character_pairings.groupByKey()

#search(adjacency_list, 'CAPTAIN AMERICA', sc)
search(adjacency_list, 'MISS THING/MARY', sc)
#search(adjacency_list, 'ORWELL', sc)

