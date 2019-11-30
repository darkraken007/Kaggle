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


def predict_using_decision_tree(x_train, y_train, x_cv, y_cv, x_test):
    # decision tree classifier
    decisionTree = DecisionTreeClassifier(criterion='gini',max_depth=10,min_samples_split=.09)
    decisionTree.fit(x_train, y_train)
    y_predict = decisionTree.predict(x_cv)

    # optimization of model
    print("score: ", decisionTree.score(x_cv, y_cv))
    print("depth: ", decisionTree.get_depth())

    # area under curve score
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_cv, y_predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("auc score: ", roc_auc)
    #CheckPerformanceWithDifferentDepth(x_train, y_train, x_cv, y_cv)
    y_test = decisionTree.predict(x_test)
    return y_test


def preprocess(x):

    x = x.drop(['PassengerId'], axis=1)
    # checking nan or null values in dataframe
    print(x.isnull().values.any())

    # handling the missing data and replace missing values with nan from numpy and replace with mean of all the other values
    imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # extracting ticket serial numbers
    x['Ticket'] = x['Ticket'].apply(convert_ticket)

    # one-hot-encoding and imputing categorical values.
    categorical_features = ['Sex', 'Embarked','Pclass']
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


def CheckPerformanceWithDifferentDepth(x_train, y_train, x_test, y_test):
    #max_depths = np.linspace(1, 32, 32, endpoint=True)
    #min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    train_results = []
    test_results = []
    for i in min_samples_leafs:
        dt = DecisionTreeClassifier(max_depth=10,min_samples_split=.09,min_samples_leaf=.3)
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

    line1, = plt.plot(min_samples_leafs, train_results, 'b' , label ="Train AUC")
    line2, = plt.plot(min_samples_leafs, test_results, 'r', label ="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()


dataset = pd.read_csv('data/train.csv', low_memory=False)
dataset_test = pd.read_csv('data/test.csv', low_memory=False)
# Splitting the attributes into independent and dependent attributes
Y = dataset['Survived']  # attributes to determine dependent variable / Class
Y = Y.to_numpy()

X = dataset.drop(['Survived', 'Cabin', 'Name'], axis=1)
X_test = dataset_test.drop(['Cabin', 'Name'], axis=1)
id = dataset_test['PassengerId']
X = preprocess(X)
X_test = preprocess(X_test)
# splitting dataset into test set and train set
# splitting the dataset into training set and test set
X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

#y_test = predict_using_decision_tree(X_train, Y_train, X_cv, Y_cv, X_test)


# Random Forest model
randomforest = RandomForestClassifier()
randomforest.fit(X_train, Y_train)
print("random forest score: ", randomforest.score(X_cv, Y_cv))
Y_test = randomforest.predict(X_test)
output1 = pd.DataFrame({'Survived': Y_test})
output1 = pd.concat([id, output1], axis=1)
output1.to_csv('outputDTC.csv', index=False)
