import numpy as np
from scipy.optimize import minimize_scalar


# Define the function
def f(x):
    return 5*np.cos(2*x) - 2*x*np.sin(2*x)


# Define the interval
interval = (1, 2)


maximum = minimize_scalar(lambda x: -f(x), bounds=interval, method='bounded')
minimum = minimize_scalar(f, bounds=interval, method='bounded')


# Print the maximum
print("Maximum value:", -maximum.fun)
print("Maximum point:", maximum.x)

# Print the minimum
print("Minimum value:", minimum.fun)
print("Minimum point:", minimum.x)


# Plot the function
import matplotlib.pyplot as plt

x = np.linspace(0, 3, 100)
plt.plot(x, f(x))
plt.plot(maximum.x, -maximum.fun, 'ro')
plt.plot(minimum.x, minimum.fun, 'go')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)', 'maximum', 'minimum'])
# draw vertical lines at the interval boundaries
plt.axvline(interval[0], color='k', linestyle='--')
plt.axvline(interval[1], color='k', linestyle='--')
plt.show()
