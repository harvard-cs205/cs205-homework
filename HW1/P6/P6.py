import pyspark
from pyspark import SparkContext, SparkConf
sc = SparkContext()



#read the file by lines
data = sc.textFile('pg100.txt')

#read the file by words
data_split = data.flatMap(lambda s: s.split())

def word_can_be_accepted(word):
    '''get rid of a word in three situations:
        contain only numbers,
        contain only letters which are capitalized,
        or contain only letters which are capitalized and end with a period.'''
    if word.isdigit():
        return False
    if word.isupper():
        return False
    if word[-1]=='.' and word[:-1].isupper():
        return False
    return True
data_check = data_split.filter(word_can_be_accepted)



#index the words for next step
data_ordered = data_check.zipWithIndex()

#index the words again 
data_ordered_agn = data_ordered.map(lambda x: (x[1], x))

#construct the first, second and third elements in the three-word pair
elements_0 = data_ordered_agn.map(lambda y: (y[0],   (y[1][0],0) )  )
elements_1 = data_ordered_agn.map(lambda y: (y[0]-1, (y[1][0],1) )  )
elements_2 = data_ordered_agn.map(lambda y: (y[0]-2, (y[1][0],2) )  )

#construct the combined elements
elements_all = sc.union([elements_0, elements_1, elements_2])



#construct the three-word pairs without them listing
three_word_pairs = elements_all.groupByKey()

#construct the three-word pairs with them showing up as list
three_word_pairs_withlist = three_word_pairs.map(lambda tpl: (tpl[0], list(tpl[1])) )

#get rid of length 1 and 2 lists produced from "groupByKey" corresponding to -1, -2,...
three_word_pairs_lengththree = three_word_pairs_withlist.filter(lambda x: len(x[1])==3)

def right_order_word_pairs(tpl):
	'''order by 0,1,2... which were assigned to words previously'''
	mylist = tpl[1]
	mylist_sorted = sorted(mylist, key = lambda x: x[1])

	mylist_cleaned = []
	for i in range(len(mylist_sorted)):
		mylist_cleaned.append(mylist_sorted[i][0])
	return mylist_cleaned
#finally, we got the form [word0, word1, word2]    
three_word_pairs_rightorder = three_word_pairs_lengththree.map(right_order_word_pairs)



#Add count 1 to each three word pair
dat = three_word_pairs_rightorder.map(lambda x: ((x[0], x[1], x[2]),1) )

#Sum up counts for same three word pairs
dat_count = dat.reduceByKey(lambda a,b: a+b)

#tuple word3 with its count
dat_word3_count = dat_count.map(lambda x: ( (x[0][0],x[0][1]) , (x[0][2],x[1]) )     )

#get something like ((Word1, Word2), [(Word3a, Count3a), (Word3b, Count3b), ...])
#but list is not showing
dat_word12_word3_count = dat_word3_count.groupByKey()

#finally get ((Word1, Word2), [(Word3a, Count3a), (Word3b, Count3b), ...])
dat_final = dat_word12_word3_count.map(lambda tpl: (tpl[0], list(tpl[1])) )





def write_phrase(rdd, num_words=20):
    #randomly choose a starting line from rdd
    starting_line = rdd.takeSample(False, 1)[0]
    
    #get word pair of this line
    starting_word_pair = starting_line[0]
    phrase = starting_word_pair[0] + ' ' + starting_word_pair[1]+ ' '
    
    #get list corresponding to this word pair
    starting_list = rdd.lookup(starting_word_pair)[0]
    starting_list_sorted = sorted(starting_list, key = lambda x: x[1])
    new_word = starting_list_sorted[0][0]
    phrase += new_word + ' '
    
    init_word_pair = (starting_word_pair[1], new_word)
    
    for i in range(num_words-3):
            #get word pair of this line
            starting_word_pair = init_word_pair
            #phrase = starting_word_pair[0] + ' ' + starting_word_pair[1]+ ' '

            #get list corresponding to this word pair
            starting_list = rdd.lookup(starting_word_pair)[0]
            starting_list_sorted = sorted(starting_list, key = lambda x: x[1])
            new_word = starting_list_sorted[0][0]
            phrase += new_word + ' '

            init_word_pair = (starting_word_pair[1], new_word)
    
    return phrase


for k in range(10):
    print write_phrase(dat_final)
    print '\n'

