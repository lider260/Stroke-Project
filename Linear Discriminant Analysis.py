import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns



from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import preprocessing
from sklearn import neighbors
import statsmodels.api as sm
import statsmodels.formula.api as smf
pd.set_option('display.width', 1000000000)
pd.set_option('display.max_columns', 100)
data = pd.read_csv('C:/Users/Derek/Documents/stroke.csv')
dfc = data.dropna()

#Linear Discriminant Analysis

X = dfc[['avg_glucose_level', 'bmi', 'hypertension']].to_numpy()
y = dfc.stroke.to_numpy()

lda = LinearDiscriminantAnalysis(solver = 'svd')
y_pred = lda.fit(X,y).predict(X)
df_ = pd.DataFrame({'True stroke status': y,
                    'Predicted stroke status' : y_pred})
df_.replace(to_replace = {0: 'No', 1: 'Yes'}, inplace = True)

print(df_.groupby(['Predicted stroke status', 'True stroke status']).size().unstack('True stroke status'))
print(classification_report(y, y_pred, target_names= ['No', 'Yes']))

#Instead of using a the probability of 50% as decision boundary, we say that a probability of default of 20% is to be classified as 'Yes'
decision_prob = 0.2
y_prob = lda.fit(X, y).predict_proba(X)

df_ = pd.DataFrame({'True stroke status': y,
                    'Predicted stroke status': y_prob[:,1] > decision_prob})
df_.replace(to_replace={0:'No', 1:'Yes', 'True':'Yes', 'False':'No'}, inplace=True)

print(df_.groupby(['Predicted stroke status','True stroke status']).size().unstack('True stroke status'))