# Practical 8: Neural Network Activation Functions
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
functions = {
    'Sigmoid': 1 / (1 + np.exp(-x)),
    'ReLU': np.maximum(0, x),
    'Tanh': np.tanh(x)
}
for name, y in functions.items():
    plt.plot(x, y, label=name)
plt.title('Activation Functions')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig('activation_functions.png')
print('Activation functions plot saved.')
