import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def pricehouse():
    plt.xkcd()
    data = pd.read_csv('linearegrssion1.csv')
    X = data.area
    y = data.price
    X = np.array([X]).T
    print(X)
    y = np.array([y]).T
    plt.title('Price of home')
    plt.xlabel('area')
    plt.ylabel('price')
    plt.scatter(X,y,marker='o',c='r')
    plt.show()

    train = LinearRegression()
    train.fit(X,y)
    w = train.coef_
    b = train.intercept_
    a = np.array([np.linspace(2500,4200,100)]).T
    ytemp = w*a+b
    plt.title('Price of home')
    plt.xlabel('area')
    plt.ylabel('price')
    plt.plot(a,ytemp)
    plt.scatter(X,y,marker='o',c='r')
    plt.show()

def percapitaincome():
    data = pd.read_csv('linearegrssion2.csv')
    X = data[['year']]
    y = data[['percapitaincome']]
    plt.xkcd()
    plt.title('Per capita income (dollar)')
    plt.xlabel("Year")
    plt.ylabel("Per capita income")
    plt.scatter(X,y,c = 'b',marker='*')
    plt.show()
    train = LinearRegression()
    train.fit(X,y)
    w = train.coef_
    b = train.intercept_
    xtemp = np.array([np.linspace(1969,2016,100)]).T
    ytemp = w*xtemp+b
    plt.title('Per capita income (dollar)')
    plt.xlabel("Year")
    plt.ylabel("Per capita income")
    plt.scatter(X,y,c = 'b',marker='*')
    plt.plot(xtemp,ytemp)
    plt.show()
#above is linear with one feature
#now is more feature
def pricehousemorefeature():
    data = pd.read_csv('linearegrssion3.csv')
    X = data[['area','bedrooms','age']]
    y = data[['price']]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=2)
    train = LinearRegression()
    train.fit(X_train,y_train)
    y_pred = train.predict(X_test)
    print(y_test)
    print(y_pred)
pricehousemorefeature()