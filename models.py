import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

def reg(features, Y_train, t_size):
    x_train, x_val, y_train, y_val = train_test_split(features, Y_train, test_size=t_size, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    print(logreg.score(x_val,y_val))
#%%
from sklearn.neural_network import MLPClassifier
def mlp(features, Y_train, t_size):
    x_train, x_val, y_train, y_val = train_test_split(features, Y_train, test_size=t_size, random_state=0)
    mlp = MLPClassifier()
    mlp.fit(x_train,y_train)
    print(mlp.score(x_val,y_val))

#%%
from sklearn.naive_bayes import MultinomialNB
def nb(features, Y_train, t_size):
    x_train, x_val, y_train, y_val = train_test_split(features, Y_train, test_size=t_size, random_state=0)
    nb = MultinomialNB
    nb.fit(x_train,y_train)
    print(nb.score(x_val,y_val))
	
train = pd.read_csv('dataset_portion.csv', sep=',')
X_train = train.Text
y_train = train.Score
