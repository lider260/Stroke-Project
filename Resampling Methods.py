import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

plt.style.use('seaborn-white')
pd.set_option('display.width', 1000000000)
pd.set_option('display.max_columns', 100)
data = pd.read_csv('C:/Users/Derek/Documents/stroke.csv')
dfc = data.dropna()

print(dfc.info())

#Cross-Validation
#t_prop = 0.5
#p_order = np.arange(1,11)
#r_state = np.arange(0,10)

#X, Y = np.meshgrid(p_order,r_state, indexing = 'ij')
#Z= np.zeros((p_order.size, r_state.size))

#regr = skl_lm.LogisticRegression()

#Generate 10 random splits of the dataset
#for (i, j), v in np.ndenumerate(Z):
#    poly = PolynomialFeatures(int(X[i, j]))
#    X_poly = poly.fit_transform(dfc.age.values.reshape(-1, 1))
#    X_train, X_test, y_train, y_test = train_test_split(X_poly, dfc.bmi.ravel(),
#                                                        test_size=t_prop, random_state=Y[i, j])
#    regr.fit(X_train, y_train)
#    pred = regr.predict(X_test)
#    Z[i, j] = mean_squared_error(y_test, pred)


#fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))

# Left plot (first split)
#ax1.plot(X.T[0],Z.T[0], '-o')
#ax1.set_title('Random split of the data set')

# Right plot (all splits)
#ax2.plot(X,Z)
#ax2.set_title('10 random splits of the data set')

#for ax in fig.axes:
#    ax.set_ylabel('Mean Squared Error')
#    ax.set_ylim(15,30)
#    ax.set_xlabel('Degree of Polynomial')
#    ax.set_xlim(0.5,10.5)
#    ax.set_xticks(range(2,11,2));

#plt.show()


p_order = np.arange(1,11)
r_state = np.arange(0,10)

# LeaveOneOut CV
regr = skl_lm.LinearRegression()
loo = LeaveOneOut()
loo.get_n_splits(dfc)
scores = list()

for i in p_order:
    poly = PolynomialFeatures(i)
    X_poly = poly.fit_transform(dfc.age.values.reshape(-1,1))
    score = cross_val_score(regr, X_poly, dfc.bmi, cv=loo, scoring='neg_mean_squared_error').mean()
    scores.append(score)


# k-fold CV
folds = 10
elements = len(dfc.index)

X, Y = np.meshgrid(p_order, r_state, indexing='ij')
Z = np.zeros((p_order.size,r_state.size))

regr = skl_lm.LinearRegression()

for (i,j),v in np.ndenumerate(Z):
    poly = PolynomialFeatures(X[i,j])
    X_poly = poly.fit_transform(dfc.age.values.reshape(-1,1))
    kf_10 = KFold(n_splits=folds, random_state=Y[i,j], shuffle = True)
    Z[i,j] = cross_val_score(regr, X_poly, dfc.bmi, cv=kf_10, scoring='neg_mean_squared_error').mean()

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))

# Note: cross_val_score() method return negative values for the scores.
# https://github.com/scikit-learn/scikit-learn/issues/2439

# Left plot
ax1.plot(p_order, np.array(scores)*-1, '-o')
ax1.set_title('LOOCV')

# Right plot
ax2.plot(X,Z*-1)
ax2.set_title('10-fold CV')

for ax in fig.axes:
    ax.set_ylabel('Mean Squared Error')
    ax.set_ylim(15,30)
    ax.set_xlabel('Degree of Polynomial')
    ax.set_xlim(0.5,10.5)
    ax.set_xticks(range(2,11,2))

plt.show()