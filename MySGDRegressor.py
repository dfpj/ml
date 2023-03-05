from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor,LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    
    def predict(self,x_test):
        li = []
        for item in x_test.values:
            re = self.a * item[0]+ self.b
            li.append(re)
        return np.array(li)





dataset = pd.read_csv("HW3/Q3/data.csv")
dataset.head()
for col in dataset.iloc[:,:1]:
    dataset[col] = dataset[col].replace(0 , np.nan)
    mean = int(dataset[col].mean(skipna=True))
    dataset[col] = dataset[col].replace(np.nan,mean)

data = dataset.iloc[:,:1]
label = dataset.iloc[:,-1:]

x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.2,random_state=0)

sgd = SGD()
sgd.fit(x_train,y_train)
y_pred = sgd.predict(x_test)
print(mean_squared_error(y_test,y_pred))

print("*" * 90)

sgd_regressor = SGDRegressor(random_state=0)
sgd_regressor.fit(x_train,y_train)
y_pred= sgd_regressor.predict(x_test)
print(mean_squared_error(y_test,y_pred))


print("*" * 90)

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
print(mean_squared_error(y_test,y_pred))

