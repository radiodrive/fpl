import numpy as np

# Inputs
x = 9.0              # e.g., "lines of code changed"
w = 0.2              # weight (importance of x)
b = -1.0             # bias

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward pass
z = w * x + b        # z = w1 * x + b
y_hat = sigmoid(z)   # ŷ = σ(z)

print("z =", z)
print("Predicted probability =", y_hat)