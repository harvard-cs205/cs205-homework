import pyspark

sc = pyspark.sparkContext()

# Read in file and transform to desired RDD form
data = sc.textFile('source.csv')
data = data.map(lambda l: l.split('"')).map(lambda l: [w for w in l if w!='' and w!=',']).map(lambda l: (l[0], l[1]))
# dimension of data tested, all are two columns.

# reduce to data2: list of (chara, [books])
data2 = data.reduceByKey(lambda a,b: a+"%$%"+b).map(lambda l: (l[0], l[1].split("%$%")))

# create graph: list of (chara, [neighbors])
data_matrix = data2.cartesian(data2)	#creates n by n array for finding neighbors
data3 = data_matrix.map(lambda l: (l[0][0], (l[0][1], l[1][0] if intersection(l[0][1],l[1][1]) and l[0][0]!=l[1][0] else '')))	#find neighbors
data3 = data3.filter(lambda l: l[1][1]!='')	# clean up matrix
data3 = data3.map(lambda l: (l[0], l[1][1]))
graph = data3.reduceByKey(lambda a,b: a+"%$%"+b).map(lambda l: (l[0], l[1].split("%$%")))


def intersection(l1,l2):
	return bool(set(l1) & set(l2))