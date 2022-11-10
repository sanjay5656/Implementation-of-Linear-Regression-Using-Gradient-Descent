# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required gr

## Program:
```

/*
Program to implement the linear regression using gradient descent.

Developed by: Sanjay S
RegisterNumber: 22007761
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("cost function using Gradienrt Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:

![ml3op1](https://user-images.githubusercontent.com/115128955/200999867-8fe3d8f5-322a-461a-92d4-1b43012d903d.png)

![ml3op2](https://user-images.githubusercontent.com/115128955/200999884-ebe5bf97-cc8a-4b97-b520-ec15572bee51.png)

![ml3op3](https://user-images.githubusercontent.com/115128955/200999910-ad9c2b80-5731-4fdb-bbcc-3a67179ff70d.png)

![ml3op4](https://user-images.githubusercontent.com/115128955/200999938-58ed5b0a-2564-4583-bef5-5402a8ab5621.png)

![ml3op5](https://user-images.githubusercontent.com/115128955/200999964-ab5c0c89-bb28-40d4-b00b-886a72b093be.png)

![ml3op6](https://user-images.githubusercontent.com/115128955/200999988-6fc4a799-e006-4c51-b33c-85789f1f5fba.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
