import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

wlist = sc.textFile('./EOWL_words.txt')

wlist2 = wlist.map(lambda line: (''.join(sorted(line)), (line)))

wlist3 = wlist2.groupByKey().mapValues(lambda x: [i for i in x ]).map(lambda y: (y[0],len(y[1]),y[1]))

wlist3.takeOrdered(1,lambda x: -x[1])