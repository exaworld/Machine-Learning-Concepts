import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')

# SUM OF SQUARED RESIDUALS
def SSR(m, b, points):
    ssr = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        ssr += (y - (m * x + b)) ** 2
    return ssr


# SUM OF SQUARED TOTALAS
def SST(points):
    sst = 0
    y_mean = points["Salary"].mean()
    for i in range(len(points)):
        y = points.iloc[i].Salary
        sst += (y - y_mean) ** 2
    return sst


# R^2
def r_2(m, b, points):
    return 1 - SSR(m, b, points) / SST(points)


# ERROR FUNCTION
def error_function(m, b, points):
    return SSR(m, b, points) / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now)) #partial derivative of error function with respect to m
        b_gradient += -(2/n) * (y - (m_now * x + b_now)) #partial derivative of error function with respect to b

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

# PRE-DEFINED VALUES
m = 9500 # Adjusted closer to the calculated value
b = 25500 # Adjusted closer to the calculated value
L = 0.01
epochs = 1000


for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)


MSE = error_function(m, b, data)
ssr = SSR(m, b, data)
sst = SST(data)
r = r_2(m, b, data)

print('MSE: ', MSE )
print('SSR: ', ssr)
print('SST: ', sst)
print('R-squared: ', r)
print('Intercept: ', b)
print('Coefficient: ', m)

plt.scatter(data.YearsExperience, data.Salary)
plt.plot(list(range(1, 11)), [m * x + b for x in range(1, 11)], color="red")
plt.show()
