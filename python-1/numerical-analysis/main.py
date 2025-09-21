import sympy as sp

from sympy.abc import x, k

# Define symbolic variable
t = sp.symbols('t')

# Define function
def f(t):
  return 1/t

F = sp.fourier_transform(sp.exp(-x**2), x, k)
print(F)
# sqrt(pi)*exp(-pi**2*k**2)

F = sp.fourier_transform(sp.exp(-x**2), x, k, noconds=False)
print(F)
# (sqrt(pi)*exp(-pi**2*k**2), True)

# Calculate Fourier Transform
F = sp.fourier_transform(f(t), t, k)
print(F)