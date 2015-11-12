def main():
	i=sc.parallelize(range(2000),10)
	j=sc.parallelize(range(2000),10)
	mandle=i.cartesian(j).map(lambda x: (x,mandelbrot(x[1]/500.0-2,x[0]/500.0-2))).cache()
	draw_image(mandle)
	plt.hist(sum_values_for_partitions(mandle).collect())