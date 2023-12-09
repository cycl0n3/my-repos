import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

# Q45
q = """
Five observations taken for two variables follow.
xi: 4   6 11  3 16
yi: 50 50 40 60 30
a. Compute and interpret the sample covariance.
b. Compute and interpret the sample correlation coefficient.
c. Develop a scatter diagram with x on the horizontal axis.
d. What does the scatter diagram developed in part (a) indicate about the relationship
between the two variables?
"""
heading(45)
xi = np.array([4, 6, 11, 3, 16], dtype=np.float32)
yi = np.array([50, 50, 40, 60, 30], dtype=np.float32)
df = pd.DataFrame({'x': xi, 'y': yi})
print(df)
sx = np.std(xi, ddof=1)
sy = np.std(yi, ddof=1)
sxy = np.cov(xi, yi, ddof=1)[0, 1]
r = np.corrcoef(xi, yi)[0, 1]
print(f'sx = {sx:.2f}, sy = {sy:.2f}, sxy = {sxy:.2f}, r = {r:.2f}')
# plt.scatter(xi, yi)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
# print('The scatter diagram shows a negative linear relationship between x and y.')

# Q46
q = """
Five observations taken for two variables follow.
xi: 6 11 15 21 27
yi: 6 9 6 17 12
a. Compute and interpret the sample covariance.
b. Compute and interpret the sample correlation coefficient.
c. Develop a scatter diagram for these data.
d. What does the scatter diagram indicate about a relationship between x and y?"""
heading(46)
xi = np.array([6, 11, 15, 21, 27], dtype=np.float32)
yi = np.array([6, 9, 6, 17, 12], dtype=np.float32)
df = pd.DataFrame({'x': xi, 'y': yi})
print(df)
sx = np.std(xi, ddof=1)
sy = np.std(yi, ddof=1)
sxy = np.cov(xi, yi, ddof=1)[0, 1]
r = np.corrcoef(xi, yi)[0, 1]
print(f'sx = {sx:.2f}, sy = {sy:.2f}, sxy = {sxy:.2f}, r = {r:.2f}')
# plt.scatter(xi, yi)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
# print('The scatter diagram shows a positive linear relationship between x and y.')

# Q47
q = """Nielsen Media Research provides two measures of the television viewing audience: a tele-
vision program rating, which is the percentage of households with televisions watching a
program, and a television program share, which is the percentage of households watching
a program among those with televisions in use. The following data show the Nielsen tele-
vision ratings and share data for the Major League Baseball World Series over a nine-year
period (Associated Press, October 27, 2003).
Rating: 19 17 17 14 16 12 15 12 13
Share: 32 28 29 24 26 20 24 20 22
--Compute and interpret the sample covariance.
--Compute the sample correlation coefficient. What does this value tell us about the
    relationship between rating and share?
--Develop a scatter diagram with rating on the horizontal axis.
--What is the relationship between rating and share? Explain.
"""
heading(47)
rating = np.array([19, 17, 17, 14, 16, 12, 15, 12, 13], dtype=np.float32)
share = np.array([32, 28, 29, 24, 26, 20, 24, 20, 22], dtype=np.float32)
df = pd.DataFrame({'rating': rating, 'share': share})
print(df)
sx = np.std(rating, ddof=1)
sy = np.std(share, ddof=1)
sxy = np.cov(rating, share, ddof=1)[0, 1]
r = np.corrcoef(rating, share)[0, 1]
print(f'sx = {sx:.2f}, sy = {sy:.2f}, sxy = {sxy:.2f}, r = {r:.2f}')
# plt.scatter(rating, share)
# plt.xlabel('rating')
# plt.ylabel('share')
# plt.show()
# print('The scatter diagram shows a positive linear relationship between rating and share.')

# Q48
q = """
A department of transportation’s study on driving speed and miles per gallon for midsize
automobiles resulted in the following data:
Speed (Miles per Hour): 30 50 40 55 30 25 60 25 50 55
Miles per Gallon: 28 25 25 23 30 32 21 35 26 25
Compute and interpret the sample correlation coefficient. What does this value tell us
"""
heading(48)
speed = np.array([30, 50, 40, 55, 30, 25, 60, 25, 50, 55], dtype=np.float32)
mpg = np.array([28, 25, 25, 23, 30, 32, 21, 35, 26, 25], dtype=np.float32)
df = pd.DataFrame({'speed': speed, 'mpg': mpg})
print(df)
sx = np.std(speed, ddof=1)
sy = np.std(mpg, ddof=1)
sxy = np.cov(speed, mpg, ddof=1)[0, 1]
r = np.corrcoef(speed, mpg)[0, 1]
print(f'sx = {sx:.2f}, sy = {sy:.2f}, sxy = {sxy:.2f}, r = {r:.2f}')
# plt.scatter(speed, mpg)
# plt.xlabel('speed')
# plt.ylabel('mpg')
# plt.show()
# print('The scatter diagram shows a negative linear relationship between speed and mpg.')

# Q50
q = """
The Dow Jones Industrial Average (DJIA) and the Standard & Poor’s 500 Index (S&P 500)
are both used to measure the performance of the stock market. The DJIA is based on the
price of stocks for 30 large companies; the S&P 500 is based on the price of stocks for 500
companies. If both the DJIA and S&P 500 measure the performance of the stock market,
how are they correlated? The following data show the daily percent increase or daily
percent decrease in the DJIA and S&P 500 for a sample of nine days over a three-month
period (The Wall Street Journal, January 15 to March 10, 2006).
DJIA: .20 .82 �.99 .04 �.24 1.01 .30 .55 �.25
S&P 500: .24 .19 �.91 .08 �.33 .87 .36 .83 �.16
--Compute the sample correlation coefficient for these data.
--Discuss the association between the DJIA and S&P 500. Do you need to check both
before having a general idea about the daily stock market performance?
--Show a scatter diagram.
"""
heading(50)
djia = np.array([.20, .82, -.99, .04, -.24, 1.01, .30, .55, -.25], dtype=np.float32)
sp500 = np.array([.24, .19, -.91, .08, -.33, .87, .36, .83, -.16], dtype=np.float32)
df = pd.DataFrame({'djia': djia, 'sp500': sp500})
print(df)
sx = np.std(djia, ddof=1)
sy = np.std(sp500, ddof=1)
sxy = np.cov(djia, sp500, ddof=1)[0, 1]
r = np.corrcoef(djia, sp500)[0, 1]
print(f'sx = {sx:.2f}, sy = {sy:.2f}, sxy = {sxy:.2f}, r = {r:.2f}')
plt.scatter(djia, sp500)
plt.xlabel('djia')
plt.ylabel('sp500')
plt.show()
# print('The scatter diagram shows a positive linear relationship between djia and sp500.')