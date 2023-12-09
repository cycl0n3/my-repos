import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

# Q25
q = """Consider a sample with data values of 10, 20, 12, 17, and 16. Compute the z-score for each
of the five observations."""
heading(25)
print('[Question]', q)
print('[Answer]')
data = [10, 20, 12, 17, 16]
print(f'data: {data}')
mean = np.mean(data)
print(f'mean: {mean}')
standard_deviation = np.std(data)
print(f'standard deviation: {standard_deviation}')
print(f'z-score: {[(x - mean) / standard_deviation for x in data]}')

# Q29
q = """The results of a national survey showed that on average, adults sleep 6.9 hours per night.
Suppose that the standard deviation is 1.2 hours.
a. Use Chebyshev’s theorem to calculate the percentage of individuals who sleep be-
tween 4.5 and 9.3 hours.
b. Use Chebyshev’s theorem to calculate the percentage of individuals who sleep be-
tween 3.9 and 9.9 hours.
c. Assume that the number of hours of sleep follows a bell-shaped distribution. Use the
empirical rule to calculate the percentage of individuals who sleep between 4.5 and
9.3 hours per day. How does this result compare to the value that you obtained using
Chebyshev’s theorem in part (a)?"""
heading(29)
print('[Question]', q)
print('[Answer]')
mean = 6.9
standard_deviation = 1.2
print(f'mean: {mean}')
print(f'standard deviation: {standard_deviation}')
a = 4.5
b = 9.3
print(f'a = {a}, b = {b}')
za = (a - mean) / standard_deviation
zb = (b - mean) / standard_deviation
print(f'z-score for a: {za}')
print(f'z-score for b: {zb}')
print(f'percentage of individuals who sleep between {a} and {b} hours: {1 - (1 / (za ** 2))}')
a = 3.9
b = 9.9
print(f'a = {a}, b = {b}')
za = (a - mean) / standard_deviation
zb = (b - mean) / standard_deviation
print(f'z-score for a: {za}')
print(f'z-score for b: {zb}')
print(f'percentage of individuals who sleep between {a} and {b} hours: {1 - (1 / (za ** 2))}')

# Q30
q = """The Energy Information Administration reported that the mean retail price per gallon of
regular grade gasoline was $2.05 (Energy Information Administration, May 2009).
Suppose that the standard deviation was $.10 and that the retail price per gallon has a bell-
shaped distribution.
a. What percentage of regular grade gasoline sold between $1.95 and $2.15 per gallon?
b. What percentage of regular grade gasoline sold between $1.95 and $2.25 per gallon?
c. What percentage of regular grade gasoline sold for more than $2.25 per gallon?"""
heading(30)
print('[Question]', q)
print('[Answer]')
mean = 2.05
standard_deviation = 0.1
print(f'mean: {mean}')
print(f'standard deviation: {standard_deviation}')
a = 1.95
b = 2.15
print(f'a = {a}, b = {b}')
za = (a - mean) / standard_deviation
zb = (b - mean) / standard_deviation
pa = (1 - (1 / (za ** 2))) / 2
pb = (1 - (1 / (zb ** 2))) / 2
print(f'z-score for a: {za}')
print(f'z-score for b: {zb}')
print(f'percentage of regular grade gasoline sold between {a} and {b} per gallon: 0.68')
a = 1.95
b = 2.25
print(f'a = {a}, b = {b}')
za = (a - mean) / standard_deviation
zb = (b - mean) / standard_deviation
print(f'z-score for a: {za}')
print(f'z-score for b: {zb}')
pa = (1 - (1 / (za ** 2))) / 2
pb = (1 - (1 / (zb ** 2))) / 2
print(f'percentage of regular grade gasoline sold between {a} and {b} per gallon: {pb + pa}')
