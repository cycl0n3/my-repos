import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

# Q38
q = """Consider a Poisson distribution with μ = 3.
a. Write the appropriate Poisson probability function.
b. Compute f(2).
c. Compute f(1).
d. Compute P(x >= 2)
"""
heading('Q38')
print(q)

μ = 3

f = lambda x: (μ**x * math.exp(-μ)) / math.factorial(x)

print('b. f(2) =', f(2))
print('c. f(1) =', f(1))
print('d. P(x >= 2) =', 1 - f(0) - f(1))

# Q40
q = """Phone calls arrive at the rate of 48 per hour at the reservation desk for Regional Airways.
a. Compute the probability of receiving three calls in a 5-minute interval of time.
b. Compute the probability of receiving exactly 10 calls in 15 minutes.
c. Suppose no calls are currently on hold. If the agent takes 5 minutes to complete the
current call, how many callers do you expect to be waiting by that time? What is the
probability that none will be waiting?
d. If no calls are currently being processed, what is the probability that the agent can take
3 minutes for personal time without being interrupted by a call?"""
heading('Q40')
print(q)

mu = 4
f = lambda x: (mu**x * math.exp(-mu)) / math.factorial(x)

print('a. P(x = 3) =', f(3))

mu = 12
f = lambda x: (mu**x * math.exp(-mu)) / math.factorial(x)

print('b. P(x = 10) =', f(10))

mu = 4
f = lambda x: (mu**x * math.exp(-mu)) / math.factorial(x)

print('c. E(x) =', mu)
print('   P(x = 0) =', f(0))

mu = 48 / 20
f = lambda x: (mu**x * math.exp(-mu)) / math.factorial(x)

print('d. P(x = 0) =', f(0))

# Q41
q = """During the period of time that a local university takes phone-in registrations, calls come in
at the rate of one every two minutes.
a. What is the expected number of calls in one hour?
b. What is the probability of three calls in five minutes?
c. What is the probability of no calls in a five-minute period?
"""
heading('Q41')
print(q)

mu = 30
f = lambda x: (mu**x * math.exp(-mu)) / math.factorial(x)

print('a. E(x) =', mu)

mu = 2.5
f = lambda x: (mu**x * math.exp(-mu)) / math.factorial(x)

print('b. P(x = 3) =', f(3))
print('c. P(x = 0) =', f(0))

# Q43
q = """Airline passengers arrive randomly and independently at the passenger-screening facility
at a major international airport. The mean arrival rate is 10 passengers per minute.
a. Compute the probability of no arrivals in a one-minute period.
b. Compute the probability that three or fewer passengers arrive in a one-minute period.
c. Compute the probability of no arrivals in a 15-second period.
d. Compute the probability of at least one arrival in a 15-second period."""
heading('Q43')
print(q)

mu = 10
f = lambda x: (mu**x * math.exp(-mu)) / math.factorial(x)

print('a. P(x = 0) =', f(0))
print('b. P(x <= 3) =', sum([f(x) for x in range(4)]))

mu = 10 / 4
f = lambda x: (mu**x * math.exp(-mu)) / math.factorial(x)

print('c. P(x = 0) =', f(0))
print('d. P(x >= 1) =', 1 - f(0))
