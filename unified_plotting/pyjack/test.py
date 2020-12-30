import pyjack
import numpy 
import matplotlib.pyplot as plt
import time

print pyjack

n = int(2e6)
A = numpy.random.rand(n) #X
#A = numpy.sin(A)
B = numpy.random.rand(n) #Y
B = numpy.sin(B)
E = 0.005 * numpy.ones(n)
C = numpy.random.rand(n)
D = numpy.random.rand(n)

ngrid = 512
lower = 0.3
upper = 1.0
kernel_n = 4.0
M_sph = 1.0
R = numpy.zeros(ngrid*ngrid)
u = numpy.ones(n)

a = time.time()
print 'start c'

output = pyjack.smoother(A, B, E, u, C, D, ngrid, lower, upper, kernel_n, M_sph)

R  = output[:ngrid**2]
Ex = output[ngrid**2:]
print(time.time()-a)

plt.imshow(R.reshape((ngrid, ngrid)))
plt.show()
