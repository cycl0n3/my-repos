import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')


# Q14 
q = "Consider a sample with data values of 10, 20, 12, 17, and 16. Compute the variance and"
"standard deviation."
heading(14)
print('[Question]', q)
print('[Answer]')
data = [10, 20, 12, 17, 16]
print(f'data: {data}')
print(f'variance: {np.var(data)}')
print(f'standard deviation: {np.std(data)}')

# Q15
q = "Consider a sample with data values of 27, 25, 20, 15, 30, 34, 28, and 25. Compute the range,"
"interquartile range, variance, and standard deviation."
heading(15)
print('[Question]', q)
print('[Answer]')
data = [27, 25, 20, 15, 30, 34, 28, 25]
print(f'data: {data}')
print(f'range: {np.ptp(data)}')
print(f'interquartile range: {np.percentile(data, 75) - np.percentile(data, 25)}')
print(f'variance: {np.var(data)}')
print(f'standard deviation: {np.std(data)}')

# Q16
q = """
A bowler’s scores for six games were 182, 168, 184, 190, 170, and 174. Using these data
as a sample, compute the following descriptive statistics:
Range
Standard deviation
Variance
Coefficient of variation"""
heading(16)
print('[Question]', q)
print('[Answer]')
data = [182, 168, 184, 190, 170, 174]
print(f'data: {data}')
print(f'range: {np.ptp(data)}')
print(f'standard deviation: %0.2f' % np.std(data))
print(f'variance: %0.2f' % np.var(data))
print(f'coefficient of variation: %0.2f' % (np.std(data)*100 / np.mean(data)))

# Q23
q = """
23. Scores turned in by an amateur golfer at the Bonita Fairways Golf Course in Bonita
Springs, Florida, during 2005 and 2006 are as follows:
2005 Season: 74 78 79 77 75 73 75 77
2006 Season: 71 70 75 77 85 80 71 79
a. Use the mean and standard deviation to evaluate the golfer’s performance over the
two-year period.
b. What is the primary difference in performance between 2005 and 2006? What im-
provement, if any, can be seen in the 2006 scores?"""
heading(23)
print('[Question]', q)
print('[Answer]')
data_2005 = [74, 78, 79, 77, 75, 73, 75, 77]
data_2006 = [71, 70, 75, 77, 85, 80, 71, 79]
print(f'data 2005: {data_2005}')
print(f'data 2006: {data_2006}')
print(f'mean 2005: {np.mean(data_2005)}')
print(f'mean 2006: {np.mean(data_2006)}')
print(f'standard deviation 2005: {np.std(data_2005)}')
print(f'standard deviation 2006: {np.std(data_2006)}')
print(f'variance 2005: {np.var(data_2005)}')
print(f'variance 2006: {np.var(data_2006)}')
print(f'coefficient of variation 2005: {np.std(data_2005)*100 / np.mean(data_2005)}')
print(f'coefficient of variation 2006: {np.std(data_2006)*100 / np.mean(data_2006)}')
