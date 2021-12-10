from statistics import mean
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# xs = np.array([1,2,4,7,4,6,9,10,8,12], dtype = np.float64)
# ys = np.array([42,45,43,47,52,46,55,57,60,58], dtype = np.float64)

def best_fit_line(xs, ys):
    xmean = mean(xs)
    ymean = mean(ys)
    x2mean = mean(xs*xs)
    xymean = mean(xs*ys)
    m = (xmean*ymean - xymean)/(xmean*xmean - x2mean)
    b = ymean - m*xmean
    return m, b


def predict(x_input):   # predicts y based on input x
    return m*x_input + b

def r2error(xs,ys,m,b):
    se1_arr = (ys - (m*xs+b))*(ys - (m*xs+b))  # array of squared error of regression line
    se2_arr = (ys - mean(ys))*(ys - mean(ys))  # array of squared error of mean line
    se1 = sum(se1_arr)
    se2 = sum(se2_arr)
    return (1-(se1/se2))

def create_dataset(n, var, step=3, correlation = 'None'):
    pivot = 4
    ys=[]
    xs=[]
    for i in range(n):
        ys.append(pivot+random.randrange(-int(var/2), int(var/2)))
        if correlation=='pos':
            pivot+=step
        elif correlation=='neg':
            pivot-=step
        xs.append(i)
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

xs, ys = create_dataset(100, 9, 0.4, 'neg')


m, b = best_fit_line(xs, ys)
r2err = r2error(xs, ys, m, b)
print(r2err)
x = np.array([i for i in range(len(xs))], dtype = np.float64)
y = m*x + b
plt.plot(x,y,'-r',label='Regression line')
# print(len(xs), len(ys))
plt.scatter(xs, ys, label='Data')
plt.legend(loc = 4)
plt.xlabel("X's")
plt.ylabel("Y's")
plt.show()


