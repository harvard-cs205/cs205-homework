import findspark
findspark.init('/home/lhoang/spark')

import pyspark
sc = pyspark.SparkContext(appName="spark1")

src = sc.textFile('source.csv')

vk = src.map(lambda x: x.split('"')).map(lambda tup: (tup[3], tup[1]))

vkj = vk.join(vk)

graph = vkj.map(lambda x: (x[1][0], x[1][1]) if (
    x[1][0] != x[1][1]) else None)
