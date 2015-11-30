import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

if __name__ == "__main__":
    wordRDD = sc.textFile("EOWL_words.txt")
    sortedWordRDD = wordRDD.map(lambda x:(x,''.join(sorted(x))))
    sortedMatchRDD = sortedWordRDD.map(lambda x:(x[1],[x[0]])).reduceByKey(lambda a,b:a+b)
    finalRDD = sortedMatchRDD.map(lambda x:(x[0],len(x[1]),x[1]))
    longest_list = finalRDD.takeOrdered(1,lambda x:-x[1])
    print longest_list
