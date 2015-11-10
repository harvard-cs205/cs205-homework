import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

from P4_bfs import ssbfs

def quiet_logs(sc):
	''' Shuts down log printouts during execution (thanks Ray!) '''
	logger = sc._jvm.org.apache.log4j
	logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)
	
def make_master_rdds(fname):
	''' Gets data from source file, creates RDD building blocks of graph representation '''
	source = sc.textFile(fname,n_parts)
	# RDD name_issue: (name, appears_in) tuples
	name_issue = (source.map(lambda x: tuple(x[1:-1].split('","')))
						.map(lambda x: (x[0].strip(),x[1]))
						.partitionBy(n_parts)
				 )

	# RDD issue_name: (appears_in, name) tuples
	issue_name = name_issue.map(lambda x:tuple(reversed(x))).partitionBy(10)

	# RDD linked_names: (appears_in, [all names that appear])
	linked_names = (name_issue.map(lambda x:tuple(reversed(x)))
							  .groupByKey() # group by issue 
							  .map(lambda x: (x[0],list(x[1]))) # convert ResultIterable to list 
							  .partitionBy(n_parts)
				   ) 
	return name_issue.cache(), issue_name.cache(), linked_names.cache()

def make_graph_rdd():
	''' Compiles graph representation: List of tuples of form (<character_name>,[<associated_names>]) '''
	# RDD: (name, [associated names]) 
	name_assoc = (issue_name.join(linked_names)
							  .map( lambda x: x[1] )
							  .groupByKey()
							  .map(lambda x: (x[0], [el for sublist in x[1] for el in sublist]))
							  .map(lambda x: (x[0], list(set([name for name in x[1] if name != x[0]]))))
							  .partitionBy(n_parts)
							  
				 )
	return name_assoc.cache()
	
quiet_logs(sc)
data_filename = "source.csv" 
n_parts = 3 # num partitions
name_issue, issue_name, linked_names = make_master_rdds(data_filename)
start_roots  = ["CAPTAIN AMERICA","MISS THING/MARY","ORWELL"]

for root in start_roots:
	name_assoc = make_graph_rdd()
	ssbfs( sc, name_assoc, root, n_parts )
 