from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np

class SGD:
    def __init__(self):
        # learning rate
        self.alpha = 0.01
        # initialze weights
        self.a = 0.5
        self.b = 1
        # checker for get coef and intercept
        self.is_finish=False

    def fit(self,x_train,y_train):
        df = pd.concat([x_train,y_train],axis=1)
        for item in df.values:
            self.get_sample(item[0],item[1],self.a,self.b)
        self.is_finish= True
    
    def get_sample(self,x,y,a,b):
        loss = self.loss_function(x,y,a,b)
        if loss**2 > 0.0001:
            a,b = self.get_new_parameter(x,y,a,b)
            self.get_sample(x,y,a,b)
        else:
            self.a = a 
            self.b = b 
            
    def get_new_parameter(self,x,y,a,b):
        a = a - (self.alpha * ((-2*a) * (y- (b + (a *x)))))
        b = b - (self.alpha * (-2 * (y- (b + (a * x)))))
        return a ,b 
    
        
    def loss_function(self,x,y,a,b):
        return y - ((a*x)+b)
    
    @property
    def coef(self):
        if self.is_finish:
            return self.a
        else:
            raise Exception("the first fit")
    
    @property
    def intercept(self):
        if self.is_finish:
            return self.b
        else:
            raise Exception("the first fit")

dataset = pd.read_csv("HW3/Q3/data.csv")
dataset.head()
for col in dataset.iloc[:,:1]:
    dataset[col] = dataset[col].replace(0 , np.nan)
    mean = int(dataset[col].mean(skipna=True))
    dataset[col] = dataset[col].replace(np.nan,mean)

data = dataset.iloc[:,:1]
label = dataset.iloc[:,-1:]



sgd = SGD()
sgd.fit(data,label)
print(sgd.coef)
print(sgd.intercept)

sgd_regressor = SGDRegressor()
sgd_regressor.fit(data,label)
print(sgd_regressor.coef_)
print(sgd_regressor.intercept_)