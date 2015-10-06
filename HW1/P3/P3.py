import numpy as np

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="P3")
words_list = sc.textFile("words")
sorted_words = words_list.map(lambda x: (''.join(sorted(x)), (1, [x])))
output = sorted_words.reduceByKey(lambda a,b: (a[0] + b[0], a[1] + b[1]))
print output.takeOrdered(10, key=lambda x: -x[1][0])