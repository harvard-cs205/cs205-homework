# Script for finding connected components in the wiki data (suitable for AWS).

import pyspark

sc = pyspark.SparkContext()

# Accessing files on AWS; note that page names and links are 1-indexed, and links dataset is 10x larger than pages
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

