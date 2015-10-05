import random
def main():
	i=sc.parallelize(range(2000),10)
	j=sc.parallelize(range(2000),10)
	mandle2=i.cartesian(j).partitionBy(100,lambda x: random.randint(0,1999)).map(lambda x: (x,mandelbrot(x[1]/500.0-2,x[0]/500.0-2))).cache()
	draw_image(mandle)
	plt.hist(sum_values_for_partitions(mandle2).collect())