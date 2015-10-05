#NOTA BENE: this code has to be copied and pasted line by line in the PySpark console of AWS, this is how I got mu result. It won't run with the commant python P3.py

#loading data from the cluster
wlist = sc.textFIle('s3://Harvard-CS205/wordlist/EOWL_words.txt')

#duplicating to form key value pairs
w_w = w.map(lambda w:(w,w))

#sorting the key
s_w = w_w.map(lambda (k,v): (''.join(sorted(k)),v))

#group all the anagrams (meaning all the word that have the same sorted word)
w_a = s_w.groupByKey().mapValues(list)

#we now have the sorted words and the lists of all corresponding anagrams, we only need to add the count
anagrame = w_a.map(lambda (k,v): (k,len(v),v))

#output the anagram with most entries
print anagrame.takeOrdered(1, lambda KV: - KV[1])
