import numpy as np
from scipy.interpolate import spline
import matplotlib.pyplot as plt
T = np.arange(100,5100,100)
print(T, len(T))
err = [159,159,158,158,158,153,153,152,152,152,152,152,151,151,
150,150,150,150,149,149,144,144,144,144,144,143,143,143,143,143,
143,143,143,143,143,143,143,143,143,143,143,141,141,141,141,
140,140,140,140,140]
print(err, len(err))

plt.figure()
xnew = np.linspace(T.min(),T.max(),300) #300 represents number of points to make between T.min and T.max

power_smooth = spline(T,err,xnew)
plt.title("Graph to compute the fitness of the best individual in the population")
# plt.plot(T,err,"go",label="best fitnesses")
plt.xlabel("iteration")
plt.ylabel("fitness value")
plt.legend(loc = "best")
plt.plot(xnew,power_smooth)

plt.show()

plt.figure()
plt.title("Graph to compute the fitness of the best individual in the population")
plt.plot(T,err, label="error rate")
# plt.plot(T,err,"go",label="best fitnesses")
plt.xlabel("iteration")
plt.ylabel("fitness value")
plt.legend(loc = "best")

plt.show() 