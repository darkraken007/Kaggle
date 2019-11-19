import numpy as np
import pandas as pd
import scipy
import math
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing as skpre
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import metrics

def data_size_response(model,trX,teX,trY,teY,score_func,prob=True,n_subsets=40):

    train_errs,test_errs = [],[]
    subset_sizes = np.exp(np.linspace(3,np.log(trX.shape[0]),n_subsets)).astype(int)

    for m in subset_sizes:
        model.fit(trX[:m],trY[:m])
        if prob:
            train_err = score_func(trY[:m],model.predict_proba(trX[:m])[:,1])
            test_err = score_func(teY,model.predict_proba(teX)[:,1])
        else:
            train_err = score_func(trY[:m],model.predict(trX[:m]))
            test_err = score_func(teY,model.predict(teX))
        train_errs.append(train_err)
        test_errs.append(test_err)

    return subset_sizes,train_errs,test_errs

def plot_response(subset_sizes,train_errs,test_errs):

    plt.plot(subset_sizes,train_errs,lw=2)
    plt.plot(subset_sizes,test_errs,lw=2)
    plt.legend(['Training Error','Test Error'])
    plt.xscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Error')
    plt.title('Model response to dataset size')
    plt.show()



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


def func_cabin(x):

    dic = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6", "G": "7", "H": "8","T": "-1"}
    if x != "-1":
        x = str(x)
        deck = x[0]
        return dic[deck]
    else:
        return "-1"


def func_preprocess(data):
    d = data;
    d = d.drop(['Name', 'Ticket', 'PassengerId','Embarked'], axis=1)
    d['Cabin'] = d['Cabin'].replace(np.NaN, "-1")
    d['Cabin'] = d['Cabin'].replace("-" "-1")
    d['Cabin'] = d['Cabin'].astype(str)
    d['Cabin'] = d['Cabin'].apply(func_cabin)
    d.loc[(d.Pclass == "1") & (d.Cabin == "-1"), 'Cabin'] = "2"
    d.loc[(d.Pclass == "2") & (d.Cabin == "-1"), 'Cabin'] = "4"
    d.loc[(d.Pclass == "3") & (d.Cabin == "-1"), 'Cabin'] = "7"
    #d = d.dropna(thresh=d.shape[0] / 2, axis=1)
    d['Sex'] = d['Sex'].apply(func_convertsex)
    #d['Embarked'] = d['Embarked'].apply(func_converboardingpoint)
    d['Age'] = d['Age'].replace(np.NaN, data['Age'].mean())
    d['SibSp'] = d['SibSp'] + d['Parch']
    d = d.drop(['Parch'],axis=1)
    # print(x.isnull().sum())
    # print(y.isnull().sum())
    return d


def standardize(x):
    x = x.to_numpy()
    sc_X = StandardScaler()
    scaler = sc_X.fit(x)
    return scaler

def func_cost(y, yhat):
    y = y.reshape((y.shape[0],1))
    yhat = yhat.reshape((yhat.shape[0], 1))
    m = y.shape[0]
    cost = (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    cost = np.sum(cost)
    return -cost/m


data_train = pd.read_csv('data/train.csv', low_memory=False)
data_test = pd.read_csv('data/test.csv', low_memory=False)

data_train = func_preprocess(data_train)
y = data_train['Survived']
y = y.to_numpy()

data_train = data_train.drop(['Survived'], axis=1);
scaler = standardize(data_train)
x = scaler.transform(data_train)
split = math.floor(x.shape[0] * .8);
x_train, x_cv = x[:split, :], x[split:, :]
y_train, y_cv = y[:split], y[split:]

logisticRegr = LogisticRegression(penalty="l1",tol=1e-10,solver="liblinear",C=.8)
sgdClass = SGDClassifier()
sgdClass.fit(x_train,y_train)
logisticRegr.fit(x_train, y_train)
print(logisticRegr.score(x_cv,y_cv))
print(sgdClass.score(x_cv,y_cv))
output1 = pd.DataFrame({'Survived': logisticRegr.predict(x_cv)})
output1.to_csv('output1.csv', index=False)
# w, costArr = lr.run_model(5, x.T, y, .00001)
# print(costArr)


# #plot learning curve
# model = logisticRegr
# score_func = func_cost
# response = data_size_response(model,x_train,x_cv,y_train,y_cv,score_func,prob=True)
# plot_response(*response)


id = data_test['PassengerId']
data_test = func_preprocess(data_test)
x_test = scaler.transform(data_test)
# Find indicies that you need to replace
inds = np.where(np.isnan(x_test))

# Place column means in the indices. Align the arrays using take
x_test[inds] = np.take(scaler.mean_, inds[1])
y_test = logisticRegr.predict(x_test)

output = pd.DataFrame({'Survived': y_test})
output = pd.concat([id, output], axis=1)
output.to_csv('output.csv', index=False)
