import numpy as np
import matplotlib.pyplot as plt


# linear function
def linear(m, b, x):
    return m * x + b


# Sum of Squared Errors
def sse(m: float, b: float, x: np.ndarray, y: np.ndarray) -> float:
    guess = linear(m, b, x)
    actual = y

    return sum((actual - guess) ** 2) / 2


# partial derivative of sse() w.r.t 'm'
def dm_sse(m: float, b: float, x: np.ndarray, y: np.ndarray):
    return sum(-x * (y - linear(m, b, x)))


# partial derivative of sse() w.r.t 'b'
def db_sse(m: float, b: float, x: np.ndarray, y: np.ndarray):
    return sum(-(y - linear(m, b, x)))


X: np.ndarray = np.linspace(0, 20, 30)
Y: np.ndarray = X + np.random.random(30) * 5 - 10

M = 0.0  # slope
B = 0.0  # intercept

# higher learning rate might cause uncontrolled bouncing & will not converge
learning_rate = 0.0001

# Gradient descent until Squared error is below 30
while sse(M, B, X, Y) > 30:
    M -= dm_sse(M, B, X, Y) * learning_rate
    B -= db_sse(M, B, X, Y) * learning_rate

print(M, B, sse(M, B, X, Y))

fitted_x = np.linspace(0, 20, 30)
fitted_y = linear(M, B, fitted_x)

plt.plot(fitted_x, fitted_y)
plt.scatter(X, Y)
plt.show()
