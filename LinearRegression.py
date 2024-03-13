# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 00:26:54 2024

@author: suvam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import load_diabetes

# data = load_diabetes()
# X = data.data
# y = data.target
# df= pd.DataFrame(X)

# print(df)
# print(X.shape)
# print(len(y))
# plt.plot(y)

X = [12,15,11,16,3,10,17,23]
y = [30,35,28,38,12,20,35,45]
coord = [(X[i],y[i]) for i in range(len(X))]
# plt.scatter(X,y)
# plt.show()

class LinearRegression():
    def __init__(self,a):
        self.a = a
        self.w = np.zeros(len(X))
        self.b = np.zeros(len(X))
    def fit(self,X,y):
        converged = False
        while converged!=True:
            J_init = self.loss(X,y)
            print("Initial Cost = ", J_init)
            w = self.w + self.a*(np.sum(np.dot(y-(np.dot(self.w,X)+self.b),X)))/len(X)
            b = self.b + self.a*(np.sum(y-(np.dot(self.w,X)+self.b)))/len(X)
            self.w = w
            self.b = b
            J_fin = self.loss(X,y)
            
            if (J_init-J_fin)**2 <0.01:
                converged = True
                print("Converged")
                break
            print("------------Interation ran-------------")
            plt.scatter(X,model.predict(np.array(X)),color='k')
            plt.scatter(X,y)
            plt.show()
            print("Final Cost = ", J_fin)
    def predict(self,X):
        return np.dot(self.w,X)+self.b
    def loss(self,X,y):
        return (np.sum((y - np.dot(X,self.w)-self.b)))/(2*len(X))
    
model = LinearRegression(0.0001)
model.fit(np.array(X),np.array(y))
X_plot = np.arange(min(X),max(X),0.1)
plt.plot(X_plot,model.predict(X_plot),color='k')
plt.scatter(X,y)
plt.show()