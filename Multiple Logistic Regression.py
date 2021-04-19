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

y= dfc.stroke
X_train = sm.add_constant(dfc[['avg_glucose_level', 'bmi', 'hypertension']])
f = 'stroke ~ avg_glucose_level + bmi + hypertension'
est= smf.logit(formula = str(f), data = dfc).fit()
print(est.summary2().tables[1])

#Balance and Default vectors for students
X_train = dfc[dfc.hypertension == 1].avg_glucose_level.values.reshape(dfc[dfc.hypertension == 1].avg_glucose_level.size,1)
y = dfc[dfc.hypertension == 1].stroke

#Balance and default vectors for non-students
X_train2 = dfc[dfc.hypertension == 0].avg_glucose_level.values.reshape(dfc[dfc.hypertension == 0].avg_glucose_level.size,1)
y2 = dfc[dfc.hypertension == 0].stroke

#Vector with balance values for plotting
X_test = np.arange(dfc.avg_glucose_level.min(), dfc.avg_glucose_level.max()).reshape(-1,1)
clf = skl_lm.LogisticRegression(solver = 'newton-cg')
clf2 = skl_lm.LogisticRegression(solver = 'newton-cg')
clf.fit(X_train,y)
clf2.fit(X_train2, y2)

prob = clf.predict_proba(X_test)
prob2 = clf2.predict_proba(X_test)
print(dfc.groupby(['hypertension','stroke']).size().unstack('stroke'))
c_palette = {'no':'lightblue', 'yes':'orange'}
# creating plot
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

# Left plot
ax1.plot(X_test, pd.DataFrame(prob)[1], color='orange', label='Hypertension')
ax1.plot(X_test, pd.DataFrame(prob2)[1], color='lightblue', label='Non-Hypertension')
ax1.hlines(60/391, colors='orange', label='Overall Hypertension',
           xmin=ax1.xaxis.get_data_interval()[0],
           xmax=ax1.xaxis.get_data_interval()[1], linestyles='dashed')
ax1.hlines(149/4309,colors='lightblue', label='Overall Non-Hypertension',
           xmin=ax1.xaxis.get_data_interval()[0],
           xmax=ax1.xaxis.get_data_interval()[1], linestyles='dashed')
ax1.set_ylabel('Stroke Rate')
ax1.set_xlabel('Average Glucose Level')
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
ax1.set_xlim(50,300)
ax1.legend(loc=2)
dfc['hypertension2'] = dfc['hypertension'].map({True: 'yes', False: 'no'})
print(dfc)
sns.boxplot(x = 'hypertension2', y =  'avg_glucose_level', data=dfc, orient='v', ax=ax2,  palette=c_palette);
plt.show()