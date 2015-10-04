import pyspark

# filter function
def filter_out_Capital(word):
    wlen=len(word)
    count=0
    for l in word:
        if l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            count=count+1

    if count==wlen:
        return False

    if count==wlen-1 and word[-1]=='.':
        return False

    return True

sc=pyspark.SparkContext()

lines=sc.textFile('Shakespeare.txt')

words=lines.flatMap(lambda line: line.split(' '))

# filter out words only containing numbers
words=words.filter(lambda word: not word.isdigit())
# filter out words only containing capital letters
words=words.filter(filter_out_Capital)

words0=words.zipWithIndex().map(lambda (w,id): (id,w))
words1=words.zipWithIndex().map(lambda (w,id): (id+1,w))
words2=words.zipWithIndex().map(lambda (w,id): (id+2,w))
words_rdd=words2.join(words1).join(words0)
words_rdd=words_rdd.map(lambda (c,((w1,w2),w3)): ((w1,w2,w3),1))
words_rdd=words_rdd.reduceByKey(lambda x,y: x+y)
words_rdd=words_rdd.map(lambda ((w1,w2,w3),c): ((w1,w2),(w3,c))).groupByKey().mapValues(list)


# generate text
# randomly take 10 samples
randWords=words_rdd.takeSample(False,10,1)
#print randWords
text_file=open('P6.txt','w')

for RW in randWords:
    randPhrase=[]
    startingWords=RW[0]
    randPhrase.append(startingWords[0])
    randPhrase.append(startingWords[1])
    loop=0
    while loop<18:
        appendWords=words_rdd.map(lambda x:x).lookup(startingWords)[0]
        #print appendWords
        apword=sc.parallelize(appendWords)
        apw=apword.sortBy(lambda aw: -aw[1]).collect()[0][0]
        startingWords=(startingWords[1],apw)
        randPhrase.append(apw)
        loop=loop+1
    # write to txt file
    randSentence=' '.join(randPhrase)
    print randSentence
    text_file.write(randSentence)
    text_file.write('\n')

text_file.close()













