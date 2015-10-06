from P4_bfs import * 
import pyspark
from pyspark import SparkContext
sc = SparkContext()

def get_hero_and_issues(data):
	# get the superhero characters
	hero_raw = data.map(lambda x: x.split('",')[0])
	heroname = hero_raw.map(lambda x: x.strip('"').encode("utf-8"))
	# get issues number
	issue_raw = data.map(lambda x: x.split('",')[1])
	issue = issue_raw.map(lambda x: x.strip('"').encode("utf-8"))
	return heroname , issue

# make a table of (issue,(hero1,hero2))
def make_table(heroname,issue):
	temp = issue.zip(heroname)
	return temp.join(temp) 

# create the graph
def create_graph(table):
	t1 = table.map(lambda x: (x[1][0],x[1][1]))
	# remove the ones whose neighbors are themselves
	graph = t1.distinct().partitionBy(8).cache()
	return graph

# read data
data = sc.textFile('source.csv')
temp = get_hero_and_issues(data)
table = make_table(temp[0],temp[1])
# create the graph
graph = create_graph(table)

table1, count1=bfs(graph, 'ORWELL',sc)
table2, count2=bfs(graph, 'MISS THING/MARY',sc)
table3, count3=bfs(graph, 'CAPTAIN AMERICA',sc)
print "ORWELL: ",count1
print "MISS THING/MARY: ", count2
print "CAPTAIN AMERICA: ", count3