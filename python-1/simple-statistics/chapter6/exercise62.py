import math
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm

from matplotlib import pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 6: Correlation')

q = """Q 8.
A random variable is normally distributed with a mean of mu = 50 and a standard devia-
tion of s = 5.
a. Sketch a normal curve for the probability density function. Label the horizontal axis
with values of 35, 40, 45, 50, 55, 60, and 65. Figure 6.4 shows that the normal curve
almost touches the horizontal axis at three standard deviations below and at three stan-
dard deviations above the mean (in this case at 35 and 65).
b. What is the probability the random variable will assume a value between 45 and 55?
c. What is the probability the random variable will assume a value between 40 and 60?
"""
heading('Q8')
print(q)

mu = 50
s = 5

# a
x = np.linspace(mu - 3*s, mu + 3*s, 100)
y = norm.pdf(x, mu, s)

# plt.plot(x, y)
# plt.show()

f = norm(mu, s)

# b
print('b. P(45 < x < 55) = ', f.cdf(55) - f.cdf(45))

# c
print('c. P(40 < x < 60) = ', f.cdf(60) - f.cdf(40))

q = """Q 11.
Given that z is a standard normal random variable, compute the following probabilities.
a. P(z <= -1.0)
b. P(z >= -1)
c. P(z >= -1.5)
d. P(-2.5 <= z)
e. P(-3 < z <= 0)
"""
heading('Q11')
print(q)

mu = 0
s = 1

f = norm(mu, s)

# a
print('a. P(z <= -1.0) = ', f.cdf(-1))

# b
print('b. P(z >= -1) = ', 1 - f.cdf(-1))

# c
print('c. P(z >= -1.5) = ', 1 - f.cdf(-1.5))

# d
print('d. P(-2.5 <= z) = ', 1 - f.cdf(-2.5))

# e
print('e. P(-3 < z <= 0) = ', f.cdf(0) - f.cdf(-3))

q = """Q 17.
For borrowers with good credit scores, the mean debt for revolving and installment
accounts is $15,015 (BusinessWeek, March 20, 2006). Assume the standard deviation is
$3540 and that debt amounts are normally distributed.
a. What is the probability that the debt for a borrower with good credit is more than
$18,000?
b. What is the probability that the debt for a borrower with good credit is less than
$10,000?
c. What is the probability that the debt for a borrower with good credit is between $12,000
and $18,000?
d. What is the probability that the debt for a borrower with good credit is no more than
$14,000?
"""
heading('Q17')
print(q)

mu = 15015
s = 3540

f = norm(mu, s)

# a
print('a. P(x > 18000) = ', 1 - f.cdf(18000))

# b
print('b. P(x < 10000) = ', f.cdf(10000))

# c
print('c. P(12000 < x < 18000) = ', f.cdf(18000) - f.cdf(12000))

# d
print('d. P(x <= 14000) = ', f.cdf(14000))

q = """Q 19.
In an article about the cost of health care, Money magazine reported that a visit to a hospi-
tal emergency room for something as simple as a sore throat has a mean cost of $328
(Money, January 2009). Assume that the cost for this type of hospital emergency room visit
is normally distributed with a standard deviation of $92. Answer the following questions
about the cost of a hospital emergency room visit for this medical service.
a. What is the probability that the cost will be more than $500?
b. What is the probability that the cost will be less than $250?
c. What is the probability that the cost will be between $300 and $400?
d. If the cost to a patient is in the lower 8% of charges for this medical service, what was
the cost of this patient’s emergency room visit?
"""
heading('Q19')
print(q)

mu = 328
s = 92

f = norm(mu, s)

# a
print('a. P(x > 500) = ', 1 - f.cdf(500))

# b
print('b. P(x < 250) = ', f.cdf(250))

# c
print('c. P(300 < x < 400) = ', f.cdf(400) - f.cdf(300))

# d
print('d. P(x < x0.08) = ', f.ppf(0.08))

q = """In January 2003, the American worker spent an average of 77 hours logged on to the Inter-
net while at work (CNBC, March 15, 2003). Assume the population mean is 77 hours, the
times are normally distributed, and that the standard deviation is 20 hours.
a. What is the probability that in January 2003 a randomly selected worker spent fewer
than 50 hours logged on to the Internet?
b. What percentage of workers spent more than 100 hours in January 2003 logged on to
the Internet?
c. A person is classified as a heavy user if he or she is in the upper 20% of usage. In
January 2003, how many hours did a worker have to be logged on to the Internet to
be considered a heavy user?
"""
heading('Q20')
print(q)

mu = 77
s = 20

f = norm(mu, s)

# a
print('a. P(x < 50) = ', f.cdf(50))

# b
print('b. P(x > 100) = ', 1 - f.cdf(100))

# c
print('c. P(x > x0.8) = ', f.ppf(0.8))
