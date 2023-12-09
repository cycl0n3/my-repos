import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

# Q15
q = """
The following table provides a probability distribution for the random variable x.
x    f(x)
3    .25
6    .50
9    .25

a. Compute E(x), the expected value of x.
b. Compute s^2, the variance of x.
c. Compute s, the standard deviation of x.
"""
heading('Q15')
print(q)

x = np.array([3, 6, 9])
f_x = np.array([.25, .50, .25])

E_x = np.sum(x * f_x)
print(f'a. E(x) = {E_x}')

s2 = np.sum((x - E_x)**2 * f_x)
print(f'b. s^2 = {s2}')

s = np.sqrt(s2)
print(f'c. s = {s}')

# Q16
q = """
The following table provides a probability distribution for the random variable y.
y    f(y)
2    .20
4    .30
7    .40
8    .10

a. Compute E(y), the expected value of y.
b. Compute s^2, the variance of y.
c. Compute s, the standard deviation of y.
"""
heading('Q16')
print(q)

y = np.array([2, 4, 7, 8])
f_y = np.array([.20, .30, .40, .10])

E_y = np.sum(y * f_y)
print(f'a. E(y) = {E_y}')

s2 = np.sum((y - E_y)**2 * f_y)
print(f'b. s^2 = {s2}')

s = np.sqrt(s2)
print(f'c. s = {s}')

# Q17
q = """
The number of students taking the Scholastic Aptitude Test (SAT) has risen to an all-time
high of more than 1.5 million (College Board, August 26, 2008). Students are allowed to
repeat the test in hopes of improving the score that is sent to college and university ad-
mission offices. The number of times the SAT was taken and the number of students are as
follows.
Number of times SAT was taken    Number of students
1                                721,769
2                                601,325
3                                166,736
4                                22,299
5                                6,730

a. Let x be a random variable indicating the number of times a student takes the SAT.
Show the probability distribution for this random variable.
b. What is the probability that a student takes the SAT more than one time?
c. What is the probability that a student takes the SAT three or more times?
d. What is the expected value of the number of times the SAT is taken? What is your in-
terpretation of the expected value?
e. What is the variance and standard deviation for the number of times the SAT is taken?
"""
heading('Q17')
print(q)

sat = np.array([1, 2, 3, 4, 5])
students = np.array([721769, 601325, 166736, 22299, 6730])

f_sat = students / np.sum(students)
print(f'a. Probability distribution for x: {f_sat}')

print(f'b. P(x > 1) = {np.sum(f_sat[1:])}')

print(f'c. P(x >= 3) = {np.sum(f_sat[2:])}')

E_x = np.sum(sat * f_sat)
print(f'd. E(x) = {E_x}')

s2 = np.sum((sat - E_x)**2 * f_sat)
print(f'e. s^2 = {s2}')
print(f'   s = {np.sqrt(s2)}')

# Q18
q = """
The American Housing Survey reported the following data on the number of bedrooms in
owner-occupied and renter-occupied houses in central cities (U.S. Census Bureau website,
March 31, 2003).
Number of bedrooms    Owner-occupied    Renter-occupied
0                     547               23
1                     5012              541
2                     6100              3832
3                     2644              8690
4 or more             557               3783

a.
Define a random variable x = number of bedrooms in renter-occupied houses and
develop a probability distribution for the random variable. (Let x = 4 represent 4 or
more bedrooms.)

b.
Compute the expected value and variance for the number of bedrooms in renter-
occupied houses.

c.
Define a random variable y = number of bedrooms in owner-occupied houses and
develop a probability distribution for the random variable. (Let y = 4 represent 4 or
more bedrooms.)

d.
Compute the expected value and variance for the number of bedrooms in owner-
occupied houses.

e.
What observations can you make from a comparison of the number of bedrooms in
renter-occupied versus owner-occupied homes?
"""
heading('Q18')
print(q)

bedrooms = np.array([0, 1, 2, 3, 4])
owner = np.array([547, 5012, 6100, 2644, 557])
renter = np.array([23, 541, 3832, 8690, 3783])

x = bedrooms
f_x = renter / np.sum(renter)
print(f'a. Probability distribution for x: {f_x}')

E_x = np.sum(x * f_x)
print(f'b. E(x) = {E_x}')

y = bedrooms
f_y = owner / np.sum(owner)
print(f'c. Probability distribution for y: {f_y}')

E_y = np.sum(y * f_y)
print(f'd. E(y) = {E_y}')

s_x = np.sqrt(np.sum((x - E_x)**2 * f_x))
s_y = np.sqrt(np.sum((y - E_y)**2 * f_y))
print(f'e. s_x = {s_x}')
print(f'   s_y = {s_y}')

print('   The number of bedrooms in renter-occupied houses is more spread out than in owner-occupied houses.')

# Q20
q = """
The probability distribution for damage claims paid by the Newton Automobile Insurance
Company on collision insurance follows.
Payment($)    Probability
0               .85
500             .04
1,000           .04
3,000           .03
5,000           .02
8,000           .01
10,000          .01

a. Use the expected collision payment to determine the collision insurance premium that
would enable the company to break even.
b. The insurance company charges an annual rate of $520 for the collision coverage.
What is the expected value of the collision policy for a policyholder? (Hint: It is the
expected payments from the company minus the cost of coverage.) Why does the
policyholder purchase a collision policy with this expected value?
"""
heading('Q20')
print(q)

payment = np.array([0, 500, 1000, 3000, 5000, 8000, 10000])
f_payment = np.array([.85, .04, .04, .03, .02, .01, .01])

E_payment = np.sum(payment * f_payment)
print(f'a. E(payment) = {E_payment}')

print(f'b. E(policy) = {E_payment - 520}')
print('   The policyholder purchases a collision policy with this expected value'
      'because it is less than the expected value of the collision policy.')

# Q21