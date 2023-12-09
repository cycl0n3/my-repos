import math
import numpy as np
import pandas as pd

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

# Q1
q = """The random variable x is known to be uniformly distributed between 1.0 and 1.5.
a. Show the graph of the probability density function.
b. Compute P(x = 1.25).
c. Compute P(1.0 <= x <= 1.25).
d. Compute P(1.20 <= x <= 1.5).
"""
heading('Q1')
print(q)

f = lambda x: x / (1.5 - 1.0)

print('b. P(x = 1.25) =', 0)
print('c. P(1.0 <= x <= 1.25) =', f(1.25) - f(1.0))
print('d. P(1.20 <= x <= 1.5) =', f(1.5) - f(1.2))

# Q2
q = """The random variable x is known to be uniformly distributed between 10 and 20.
a. Show the graph of the probability density function.
b. Compute P(x <= 15).
c. Compute P(12 <= x <= 18).
d. Compute E(x).
e. Compute Var(x)
"""
heading('Q2')
print(q)

f = lambda x: (x - 10) / (20 - 10)

print('b. P(x <= 15) =', f(15))
print('c. P(12 <= x <= 18) =', f(18) - f(12))
print('d. E(x) =', (10 + 20) / 2)
print('e. Var(x) =', (20 - 10)**2 / 12)

# Q3
q = """Delta Airlines quotes a flight time of 2 hours, 5 minutes for its flights from Cincinnati to
Tampa. Suppose we believe that actual flight times are uniformly distributed between
2 hours and 2 hours, 20 minutes.
a. Show the graph of the probability density function for flight time.
b. What is the probability that the flight will be no more than 5 minutes late?
c. What is the probability that the flight will be more than 10 minutes late?
d. What is the expected flight time?"""
heading('Q3')
print(q)

f = lambda x: (x - 120) / (140 - 120)

print('b. P(x <= 125) =', f(125))
print('c. P(x >= 130) =', 1 - f(130))
print('d. E(x) =', (120 + 140) / 2)