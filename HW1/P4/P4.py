
# coding: utf-8

# In[1]:

import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import re
from P4_bfs import bfs


# In[2]:

#source_rdd will contain (key=character, value=a comic that the character is in)
source_rdd = sc.textFile("source.csv",100)
cleaner_regex = re.compile('"(.+)","(.+)"')
source_rdd = source_rdd.map(lambda line: cleaner_regex.search(line).groups())


# In[3]:

#comic_rdd will contain (key=comic, value=list of characters in the comic)
comic_rdd = source_rdd.map(lambda (character, comic): (comic, [character]))
comic_rdd = comic_rdd.reduceByKey(lambda chars1, chars2: chars1 + chars2)


# In[4]:

#char_rdd will contain (key=character, value=set of comics the character is in)
char_rdd = source_rdd.map(lambda (character, comic): (character, set([comic])))
char_rdd = char_rdd.reduceByKey(lambda chars1, chars2: chars1.union(chars2))


# In[5]:

#now we want to make adj_matrix_rdd which contains 
#(key=character1, value=set of characters that appear in some comic with character1)

#dictionary where key=comic, value=list of characters in the comic
comic_dict = comic_rdd.collectAsMap()

def flatten (lst_of_lsts):
    #helper function that takes a list of list and returns a flattened list
    flat = []
    for l in lst_of_lsts:
        flat.extend(l)
    return flat

adj_matrix_rdd = char_rdd.map(lambda (char,comics): (char, list(set(flatten([comic_dict[comic] for comic in comics])))))
adj_matrix_rdd = adj_matrix_rdd.partitionBy(100)
adj_matrix_rdd.cache()


# In[ ]:

#BFS using RDDs
sources = ['CAPTAIN AMERICA','MISS THING/MARY','ORWELL']
#call into bfs
for source in sources:
    bfs(sc, adj_matrix_rdd, source, 10)


# In[ ]:



