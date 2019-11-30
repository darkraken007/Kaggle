# libraries
import numpy as np  # used for handling numbers
import pandas as pd  # used for handling the dataset
from sklearn.impute import SimpleImputer  # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # used for encoding categorical data
from sklearn.model_selection import train_test_split  # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler  # used for feature scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def convert_ticket(x):
    return x.split(" ")[-1]


def func_cabin(x):
    dic = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6", "G": "7", "H": "8", "T": "-1"}
    if x != "-1":
        x = str(x)
        deck = x[0]
        return dic[deck]
    else:
        return "-1"


def preprocess(x):
    # x['Cabin'] = x['Cabin'].replace(np.NaN, "-1")
    # x['Cabin'] = x['Cabin'].replace("-" "-1")
    # x['Cabin'] = x['Cabin'].astype(str)
    # x['Cabin'] = x['Cabin'].apply(func_cabin)
    # x.loc[(x.Pclass == "1") & (x.Cabin == "-1"), 'Cabin'] = "2"
    # x.loc[(x.Pclass == "2") & (x.Cabin == "-1"), 'Cabin'] = "4"
    # x.loc[(x.Pclass == "3") & (x.Cabin == "-1"), 'Cabin'] = "7"
    x = x.drop(['PassengerId'], axis=1)
    # checking nan or null values in dataframe
    print(x.isnull().values.any())

    # handling the missing data and replace missing values with nan from numpy and replace with mean of all the other values
    imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # extracting ticket serial numbers
    x['Ticket'] = x['Ticket'].apply(convert_ticket)

    # one-hot-encoding and imputing categorical values.
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    categorical_transformer = Pipeline(steps=[
        ('imputer', imputer_median),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # imputing and scaling
    numeric_features = ['Age']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    # using column transformer to apply transformers defined above
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    x = preprocessor.fit_transform(x)

    return x


def CheckPerformance(x_train, y_train, x_test, y_test):
    num_forests = np.linspace(5, 100, 20, endpoint=True, dtype=int)
    train_results = []
    test_results = []
    for i in num_forests:
        dt = RandomForestClassifier(n_estimators=i)
        dt.fit(x_train, y_train)
        train_pred = dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous train results
        train_results.append(roc_auc)
        y_pred = dt.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous test results
        test_results.append(roc_auc)

    line1, = plt.plot(num_forests, train_results, 'b', label="Train AUC")
    line2, = plt.plot(num_forests, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()


dataset = pd.read_csv('data/train.csv', low_memory=False)
dataset_test = pd.read_csv('data/test.csv', low_memory=False)
# Splitting the attributes into independent and dependent attributes
Y = dataset['Survived']  # attributes to determine dependent variable / Class
Y = Y.to_numpy()
dataset.dropna(thresh=2)
X = dataset.drop(['Survived', 'Cabin', 'Name'], axis=1)
X_test = dataset_test.drop(['Cabin', 'Name'], axis=1)
id = dataset_test['PassengerId']
X = preprocess(X)
X_test = preprocess(X_test)
# splitting dataset into test set and train set
# splitting the dataset into training set and test set
X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
X_train1, X_cv1, Y_train1, Y_cv1 = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

CheckPerformance(X_train, Y_train, X_cv, Y_cv)

# Random Forest model
randomforest = RandomForestClassifier()
randomforest.fit(X_train, Y_train)
print("random forest score: ", randomforest.score(X_cv, Y_cv))
Y_test = randomforest.predict(X_test)
output1 = pd.DataFrame({'Survived': Y_test})
output1 = pd.concat([id, output1], axis=1)
output1.to_csv('outputDTC.csv', index=False)
