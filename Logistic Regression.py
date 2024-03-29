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


data = pd.read_csv('C:/Users/Derek/Documents/stroke.csv')
dfc = data.dropna()
print(dfc.columns)
X_train = dfc.avg_glucose_level.values.reshape(-1,1)
y = dfc.stroke

#Create array of test Data. Calculate the classification probability # and predicted classification

X_test = np.arange(dfc.avg_glucose_level.min(), dfc.avg_glucose_level.max()).reshape(-1,1)

clf = skl_lm.LogisticRegression(solver = 'newton-cg')
clf.fit(X_train,y)
prob = clf.predict_proba(X_test)

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,5))
#Left plot
sns.regplot(dfc.avg_glucose_level, dfc.stroke, order =1, ci = None, scatter_kws= {'color': 'orange'},
            line_kws = {'color': 'lightblue', 'lw':2}, ax=ax1)
#Right Plot
ax2.scatter(X_train, y, color = 'orange')
ax2.plot(X_test, prob[:,1], color = 'lightblue')

for ax in fig.axes:
    ax.hlines(1, xmin=ax.xaxis.get_data_interval()[0],
              xmax=ax.xaxis.get_data_interval()[1], linestyles='dashed', lw=1)
    ax.hlines(0, xmin=ax.xaxis.get_data_interval()[0],
              xmax=ax.xaxis.get_data_interval()[1], linestyles='dashed', lw=1)
    ax.set_ylabel('Probability of a stroke')
    ax.set_xlabel('Glucose Level')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.])
    ax.set_xlim(xmin=0)
plt.show()

#Using Newton-cg solver, the coefficients are equal/closest to the ones in the book
y=dfc.stroke
clf = skl_lm.LogisticRegression(solver = 'newton-cg')
X_train = dfc.avg_glucose_level.values.reshape(-1,1)
clf.fit(X_train, y)
print(clf)
print('classes: ', clf.classes_)
print('coefficients: ', clf.coef_)
print('intercept: ', clf.intercept_)

X_train = sm.add_constant(dfc.avg_glucose_level)
est = smf.logit(y.ravel(), X_train).fit()
est.summary2().tables[1]

