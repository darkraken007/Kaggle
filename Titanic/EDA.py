# Import modules
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Figures inline and set visualization style
#%matplotlib inline
sns.set()



df_train = pd.read_csv('data/train.csv', low_memory=False)
df_test = pd.read_csv('data/test.csv', low_memory=False)
# Splitting the attributes into independent and dependent attributes
Y = df_train['Survived']  # attributes to determine dependent variable / Class
Y = Y.to_numpy()
df_train.dropna(thresh=2)
# View first lines of test data

sns.countplot(x='Survived', data=df_train)
sns.countplot(x='Sex', data=df_train)
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train)
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);

df_train.groupby('Survived').Fare.hist(alpha=0.6);
sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.3, jitter=True);
sns.swarmplot(x='Survived', y='Fare', data=df_train);
sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.5});