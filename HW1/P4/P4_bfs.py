import findspark 
findspark.init()
from pyspark import SparkContext

sc = SparkContext()

#sudo-ish coding because ran out of time to work on it
def bfs(character, graph): 
    #Things to be aware of
    #   make sure there is a way to prevent infinite loops a->b->a->b->...
    #       mark each searched value with true or false? 
    #           true if searched already, false if not
    #   check list of values for character
    #   if not found, use those characters as key and check their values
    #       (this will branch extensively, use partitions to break up work)
    #   if character is found, backtrack by returning the character name
    #return a list of characters that the node was connected to
