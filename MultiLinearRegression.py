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

X = np.array([(12,112),(15,115),(11,111),(16,116),(3,103),(10,100),(17,117),(23,123)])
y = [30,35,28,38,12,20,35,45]
# plt.scatter(X,y)
# plt.show()

class LinearRegression():
    def __init__(self,a,n_iter):
        self.a = a
        self.n_iter = n_iter
        self.w = None
        self.b = 0
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.w = np.random.rand(n_features)
        for i in range(self.n_iter+1):
            J_init = self.loss(X,y)
            print("Initial Cost = ", J_init)
            y_pred = np.dot(X,self.w)+self.b
            
            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)
            
            self. w = self.w - self.a*dw
            self.b = self.b - self.a*db
            
            #w = self.w + self.a*(np.sum(np.dot(y-(self.w*X+self.b),X)))/len(X)
            #b = self.b + self.a*(np.sum(y-(self.w*X+self.b)))/len(X)
            #self.w = w
            #self.b = b
            J_fin = self.loss(X,y)
        
            # print("------------Interation ran-------------")
            # plt.scatter(X[:,0],model.predict(np.array(X)),color='k')
            # plt.scatter(X[:,0],y)
            # plt.show()
            print("Final Cost = ", J_fin)
            
            if i>500:
                self.a = self.a*(db)
    def predict(self,X):
        return np.dot(X,self.w)+self.b
    def loss(self,X,y):
        return np.sum(np.dot(X,self.w)+self.b - y)/(2*len(X))
    #(np.sum((np.dot(X,self.w)-self.b)))/(2*len(X))
    
model = LinearRegression(0.0001,1500)
model.fit(np.array(X),np.array(y))
# X_plot = np.arange(min(X[:,0]),max(X[:,0]),0.1)
# plt.plot(X_plot,model.predict(X_plot),color='k')
# plt.scatter(X[:,0],y)


# plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = X[:,0]
ys = X[:,1]    
zs = y
z_pred = model.predict(X)
ax.scatter(xs, ys, zs, marker='o')
ax.scatter(xs, ys, z_pred, marker='X')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()