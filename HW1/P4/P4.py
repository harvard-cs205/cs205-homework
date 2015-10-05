import findspark
findspark.init('/Users/Grace/spark-1.5.0-bin-hadoop2.6/')
import pyspark
import copy
from P4_bfs import *

sc = pyspark.SparkContext()
raw_data = sc.textFile("source.csv")


def formatting(row):
    #input will be (character_name, book)
    pair = row.split('","')
    character = pair[0].replace('"', '')
    book = pair[1].replace('"', '')
    return (book, [character])

def character_and_neighbor(groupby_book_tuple):
    #input will be (book, [characters_belong_to_this_book])
    #return [(character1, [neighbors_without_this_character1]), (character2, [neighbors_without_character2])... ]
    
    characters_list = groupby_book_tuple[1]
    #book = groupby_book_tuple[0]
    result = []

    for character in characters_list:
        neighbors_without_this_character = copy.deepcopy(characters_list)
        neighbors_without_this_character.remove(character)
        result.append((character, neighbors_without_this_character))
    
    return tuple(result)


#split_rdd = raw_data.map(lambda x: x.split('","')).map(lambda (character, book) : (book.replace('"', ''), character.replace('"', '')))
data_rdd = raw_data.map(formatting)
#print data_rdd.take(1)
groupby_book = data_rdd.reduceByKey(lambda x, y : x + y)
#print groupby_book.take(1)
not_refined_graph = groupby_book.flatMap(character_and_neighbor)
#print not_refined_graph.take(1)
marvel_character_graph = not_refined_graph.reduceByKey(lambda book1_neighbors, book2_neighbors : list(set(book1_neighbors + book2_neighbors)))
#print len(marvel_character_graph.take(1)[0][1])


marvel_character_graph.cache()

print BFS_relax_diameter(marvel_character_graph, 'ORWELL', sc)
print BFS_relax_diameter(marvel_character_graph, 'MISS THING/MARY', sc)
print BFS_relax_diameter(marvel_character_graph, 'CAPTAIN AMERICA', sc)
