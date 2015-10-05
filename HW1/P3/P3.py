import findspark
findspark.init('/home/toby/spark')

import pyspark

if __name__ == '__main__':
    sc = pyspark.SparkContext(appName="Spark 2")
    sc.setLogLevel('WARN') 
    myfiles = "./words/*.csv"
    words = sc.textFile(myfiles) # read files

    words = words.map(lambda word: ["".join(sorted((word))), [word]]) # add sorted word as key
    words = words.reduceByKey(lambda a, b: a+b) # reduce by sorted word

    words = words.map(lambda element: [element[0], len(element[1]), element[1]])
    print words.takeOrdered(1, key=lambda x: -x[1]) # print out most frequent one
