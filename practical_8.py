# Practical 8: Neural Network Activation Functions
# Plots Sigmoid, ReLU, and Tanh activation functions.

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate input values from -5 to 5
x = np.linspace(-5, 5, 100)

# 2. Define activation functions
sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)
tanh = np.tanh(x)

# 3. Plot the activation functions
plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid, label="Sigmoid", color="blue", linewidth=2)
plt.plot(x, relu, label="ReLU", color="green", linewidth=2)
plt.plot(x, tanh, label="Tanh", color="red", linewidth=2)
plt.title("Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("activation_functions.png")
print("Activation functions plot saved as 'activation_functions.png'.")
