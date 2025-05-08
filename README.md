# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: HIRUTHIK SUDHAKAR
RegisterNumber: 212223240054
*/
```
```python

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```
## Output:
### Data Head:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/b2f6f2eb-1e0c-4fbb-8784-4a8bd706c979)
### Data Info:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/7c13b486-2ad5-4e1f-82f6-48d7f77e2649)

### isnull() sum():
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/3a21fac0-df89-4aaf-827f-bc00aa3f0286)
### Data Head for salary:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/0a79abfa-f32d-4394-a73d-47161eaeec30)

### Mean Squared Error :
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/3c7acf12-adb7-4a3f-807e-cb49ad260032)

### r2 Value:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/e6f5cab9-dab9-4c69-bb0e-6fa0abee1da0)

### Data prediction :

![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/92b5c1d6-e495-4eaa-9a9a-8eb3a37ae0bc)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
