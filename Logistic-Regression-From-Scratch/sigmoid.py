import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-10,10,1000)
y = 1/(1+np.exp(-x))


plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(True)
plt.show()