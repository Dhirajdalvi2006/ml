import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

plt.plot(x, x, label="Linear")
plt.plot(x, 1/(1+np.exp(-x)), label="Sigmoid")
plt.plot(x, np.maximum(0, x), label="ReLU")
plt.plot(x, np.tanh(x), label="Tanh")

plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.show()