import pyspark
from P4_bfs import Graph_BFS

sc = pyspark.SparkContext('local','Graph_BFS')

lines=sc.textFile('source.csv')
lines=lines.map(lambda l: l.split(','))

# clean up, seperate the hero names and comic series
lines_sp=lines.map(lambda l: (''.join(l[:-1]),l[-1]))
tuple_heros=lines_sp
tuple_comics=lines_sp.map(lambda l: (l[1], l[0]))

# two rdds with hero as key and comics as key respectively
hero_key_group=tuple_heros.groupByKey().mapValues(list)
comics_key_group=tuple_comics.groupByKey().mapValues(list)

# product them to get a larger table
L_table=hero_key_group.cartesian(comics_key_group)

# filter by comics
L_table_filtered_comics=L_table.filter(lambda LT: LT[1][0] in LT[0][1])

# get rid of comics, only heros...
table_heros=L_table_filtered_comics.map(lambda TFC: (TFC[0][0], TFC[1][1]))

# then group by hero
hero_graph=table_heros.reduceByKey(lambda x,y: x+y)
# get rid of duplications
hero_graph=hero_graph.mapValues(set)
hero_graph=hero_graph.mapValues(list)
# filter out the key hero in the neighbor list
hero_graph=hero_graph.map(lambda (K,V): (K, [x for x in V if x!=K]))

# try out these strings
v_str1='"CAPTAIN AMERICA"'
v_str2='"MISS THING/MARY"'
v_str3='"ORWELL"'

Graph_BFS(hero_graph,v_str1,0,sc)
