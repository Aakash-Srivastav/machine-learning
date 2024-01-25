import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def variables():
    global housing
    global X
    global Y
    global X_train,X_test,Y_train,Y_test

    housing = pd.read_excel('ASSIGNMENT_1.xlsx')
    housing = housing.drop('Transaction date',axis=1)
    X = housing.drop(['House price of unit area'],axis=1)
    Y = housing['House price of unit area']
    X_train , X_test , Y_train , Y_test = train_test_split(
        X , Y , train_size=0.7 , test_size=0.3 , random_state=0
    )

def Linear_Regression_Model():
    """
    1) This function will call the Linear Regression Model
    2) Train the Model with train data 
    3) Then predict on test data
    4) Finally it will calculate error between the actual value of test data and predicted value of test data
    """
    model_LR = LinearRegression()
    model_LR.fit(X_train, Y_train)
    Y_pred = model_LR.predict(X_test)
    error = mean_absolute_percentage_error(Y_test, Y_pred)
    return(f'Error Percentage of Linear Regression Model is {error}')

def Support_Vector_Machine_Model():
    """
    1) This function will call the Support Vector Regression Model
    2) Train the Model with train data 
    3) Then predict on test data
    4) Finally it will calculate error between the actual value of test data and predicted value of test data    
    """
    model_SVR = svm.SVR()
    model_SVR.fit(X_train, Y_train)
    Y_pred = model_SVR.predict(X_test)
    error = mean_absolute_percentage_error(Y_test, Y_pred)
    return(f'Error Percentage of Support Vector Regression Model is {error}')

def Random_Forest_Regression_Model():
    """
    1) This function will call the Random Forest Regression Model
    2) Train the Model with train data 
    3) Then predict on test data
    4) Finally it will calculate error between the actual value of test data and predicted value of test data    
    """
    model_RFR = RandomForestRegressor(n_estimators=10)
    model_RFR.fit(X_train, Y_train)
    Y_pred = model_RFR.predict(X_test)
    error = mean_absolute_percentage_error(Y_test, Y_pred)
    return(f'Error Percentage of Random Forest Regression Model is {error}')

if "__main__" == __name__:
    Linear_Regression_Model()
    Support_Vector_Machine_Model()
    Random_Forest_Regression_Model()