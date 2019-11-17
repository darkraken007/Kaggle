import numpy as np
import pandas as pd
import scipy
import math
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as skpre
from sklearn.preprocessing import StandardScaler


# male is 1 female is 0
def func_convertsex(x):
    if x == 'male':
        return 1;
    else:
        return 2;


# Cherbourg - 1 Queenstown - 2 Southampton - 3
def func_converboardingpoint(x):
    if x == 'C':
        return 1
    elif x == "Q":
        return 2
    else:
        return 3


def func_preprocess(data):
    d = data;
    d = d.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
    d = d.dropna(thresh=d.shape[0]/2, axis=1)
    d['Sex'] = d['Sex'].apply(func_convertsex)
    d['Embarked'] = d['Embarked'].apply(func_converboardingpoint)
    d['Age'] = d['Age'].replace(np.NaN, data['Age'].mean())
    # print(x.isnull().sum())
    # print(y.isnull().sum())
    return d

def standardize(x):

    x = x.to_numpy()
    sc_X = StandardScaler()
    scaler = sc_X.fit(x)
    return scaler


data_train = pd.read_csv('data/train.csv', low_memory=False)
data_test = pd.read_csv('data/test.csv', low_memory=False)

data_train = func_preprocess(data_train)
y = data_train['Survived']
y = y.to_numpy()
data_train = data_train.drop(['Survived'], axis=1);
scaler = standardize(data_train)
x = scaler.transform(data_train)
split = math.floor(x.shape[0]*.8);
x_train, x_cv = x[:split, :], x[split:, :]
y_train, y_cv = y[:split], y[split:]

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
print(logisticRegr.score(x_cv, y_cv))
# w, costArr = lr.run_model(5, x.T, y, .00001)
# print(costArr)
id = data_test['PassengerId']
data_test = func_preprocess(data_test)
print(data_test)
x_test = scaler.transform(data_test)
#Find indicies that you need to replace
inds = np.where(np.isnan(x_test))

#Place column means in the indices. Align the arrays using take
x_test[inds] = np.take(scaler.mean_, inds[1])
print(logisticRegr.score(x_cv, y_cv))
y_test = logisticRegr.predict(x_test)

output = pd.DataFrame({'Survived':y_test})
output = pd.concat([id, output],axis=1)
output.to_csv('output.csv', index=False)
