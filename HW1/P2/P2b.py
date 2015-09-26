from P2 import *

pix1D=sc.parallelize(pixel,10)
pixels = pix1D.cartesian(pix1D)
mandrdd = pixels.map(lambda i: (i, mandelbrot((i[1]/500.)-2,(i[0]/500.0)-2)))
draw_image(mandrdd)
plt.hist(sum_values_for_partitions(mandrdd).collect())
plt.savefig("P2a_hist.png")
plt.xlabel("Iteration Count")
plt.ylabel("Worker Count")

range2000 = range(2000)
def randomList(a):
    b = []
    for i in range(len(a)):
        element = random.choice(a)
        a.remove(element)
        b.append(element)
    return b

randomlist = randomList(range2000)
pix1Drand = sc.parallelize(randomlist,10)
pixels2 = pix1Drand.cartesian(pix1Drand)

mandrdd2 = pixels2.map(lambda i: (i, mandelbrot((i[1]/500.)-2,(i[0]/500.0)-2)))

plt.hist(sum_values_for_partitions(mandrdd2).collect())
plt.savefig("P2b_hist.png")
plt.xlabel("Iteration Count")
plt.ylabel("Worker Count")