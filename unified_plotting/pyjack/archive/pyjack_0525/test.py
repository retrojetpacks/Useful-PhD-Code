import pyjack
import numpy 
import matplotlib.pyplot as plt
import time


n = int(2e6)
A = numpy.random.rand(n)
#A = numpy.sin(A)
B = numpy.random.rand(n)
B = numpy.sin(B)
E = 0.005 * numpy.ones(n)
ngrid = 512
lower = 0.3
upper = 1.0
kernel_n = 4.0
M_sph = 1.0
R = numpy.zeros(ngrid*ngrid)

a = time.time()
print 'start c'
R = pyjack.smoother(A, B, E, ngrid, lower, upper, kernel_n, M_sph)
print(time.time()-a)

plt.imshow(R.reshape((ngrid, ngrid)))
plt.show()