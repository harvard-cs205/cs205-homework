import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()


words = sc.textFile('shakespeare.txt').flatMap(lambda x: x.split()).filter(lambda x: x.upper != x)

