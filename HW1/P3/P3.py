import seaborn as sns
sns.set_context('poster', font_scale=1.25)
import findspark as fs
fs.init()
import pyspark as ps
import multiprocessing as mp
import string

### Setup ###
config = ps.SparkConf()
config = config.setMaster('local[' + str(2*mp.cpu_count()) + ']')
config = config.setAppName('anagram_solver')
sc = ps.SparkContext(conf=config)

### Parsing the data efficiently! ###
wlist = sc.textFile('EOWL_words.txt', use_unicode=True)

def get_anagrams_key(input_str):
    alphabet = string.ascii_lowercase
    key = [0 for z in range(len(alphabet))]
    for count, cur_letter in enumerate(alphabet):
        key[count] = input_str.count(cur_letter)
    key = tuple(key)
    return (key, input_str)

key_string_rdd = wlist.map(get_anagrams_key)
grouped_by_anagram_rdd = key_string_rdd.groupByKey()

def get_into_final_form(x):
    words = list(x[1])
    num_anagrams = len(words)
    sorted_key = ''.join(sorted(words[0]))
    return (sorted_key, num_anagrams, words)

final_form_rdd = grouped_by_anagram_rdd.map(get_into_final_form)
final_form_result = final_form_rdd.collect()

# Sort the final form data by number of anagrams
final_form_result.sort(key=lambda x: x[1], reverse=True)

# Print the first 5 top hits to prove that my script works!
print final_form_result[0:5]