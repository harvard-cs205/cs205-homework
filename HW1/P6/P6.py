import findspark
findspark.find()
findspark.init('/usr/local/opt/apache-spark/libexec')
import pyspark
import copy

sc = pyspark.SparkContext()

def getWordCount(wlist):
    result = {}
    for i in wlist:
        try:
            result[i] = result[i] + 1
        except: 
            result[i] = 1
    result = sorted(result.items(), key=lambda x: x[1], reverse = True)
    return result

def generateLine(word_count):
    init_key = resultRDD.keys().takeSample(True, 1)[0] 
    phrase = init_key[0] + ' ' + init_key[1]
    for i in xrange(word_count - 2):
        try:
            new_word = resultRDD.map(lambda x: x).lookup(init_key)[0][0][0]
        except: 
            init_key = resultRDD.keys().takeSample(True, 1)[0]
            new_word = resultRDD.map(lambda x: x).lookup(init_key)[0][0][0]
        
        phrase = phrase + ' ' + new_word
        init_key = (init_key[1], new_word)
    
    return phrase
    
def multipleLines(line_count, word_count):
    lines = ''
    for i in xrange(line_count):
        lines = lines + '\n' + generateLine(word_count)
    return lines

if __name__ == '__main__':

    rawData = sc.textFile('pg100.txt')
    wlist = rawData.flatMap(lambda x: x.split(' ')).filter(lambda x:
        not x.isdigit()).filter(lambda x:
        not x.isupper()).filter(lambda x:
        x != '')

    wlist_2 = wlist.collect()
    word_seq = []
    for i in xrange(len(wlist_2)-2):
        word_seq.append(((wlist_2[i], wlist_2[i+1]),wlist_2[i+2]))

    resultRDD = sc.parallelize(word_seq).groupByKey().mapValues(lambda x: list(x)).mapValues(getWordCount).cache()

    Shakespeare_verse = multipleLines(10, 20)

    print Shakespeare_verse
