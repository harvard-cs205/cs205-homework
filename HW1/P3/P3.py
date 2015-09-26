import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="cs205")

def get_wlist_rdd():
    import boto
    from boto.s3.connection import OrdinaryCallingFormat
    cred_file = open('/Users/andrew/Google Drive/AWS/credentials.csv', 'r')
    creds = cred_file.read().split("\n")[1].split(",")
    access_key = creds[1]
    secret_key = creds[2]
    cred_file.close()
    conn = boto.connect_s3( 
        aws_access_key_id = access_key, 
        aws_secret_access_key = secret_key, 
        calling_format = OrdinaryCallingFormat() # allows for capital letters in bucket name (non-standard)
    )
    # get word list from s3
    wlist = conn.get_bucket("Harvard-CS205").get_key('wordlist/EOWL_words.txt').get_contents_as_string().split("\n")  
    return sc.parallelize(wlist,10)

def find_anagrams(data,master):
    '''Gets all permutations of word, finds valid cases by comparing against master word list'''
    from itertools import permutations
    perms = set([''.join(p) for p in permutations(data[0])])
    valid_perms = list(perms.intersection(master))
    return data + (valid_perms,)

def solution(local=True):
    ''' Prints anagram solution.'''
    ''' NOTE: Set local=False if running on AWS'''
    rdd = get_wlist_rdd() if local else sc.textFile('s3://Harvard-CS205/wordlist/EOWL_words.txt',150)
    return find_anagrams(
            rdd.map(sorted) \
            .map(lambda x: (''.join(x),1)) \
            .reduceByKey(lambda a,b: a+b) \
            .takeOrdered(1, key=lambda x: -x[1])[0],
            rdd.collect()
          )
    
print solution()