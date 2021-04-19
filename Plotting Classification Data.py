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

dfc['gender2'] = dfc.gender.factorize()[0]


#Plotting the Categorial Data

fig = plt.figure(figsize= (12,5))
gs = mpl.gridspec.GridSpec(1,4)
ax1 = plt.subplot(gs[0,:-2])
ax2 = plt.subplot(gs[0,-2])
ax3 = plt.subplot(gs[0,-1])

#Take a fraction of the samples where target value (stroke) is 'no'
df_no = dfc[dfc.stroke == 0].sample(frac= 0.15)
#Take all samples where target value is yes
df_yes = dfc[dfc.stroke == 1]
df_ = df_no.append(df_yes)

ax1.scatter(df_[df_.stroke == 1].bmi, df_[df_.stroke == 1].age, s = 40, c = 'orange', marker = '+'
            , linewidths = 1, label = 'Had Stroke')
ax1.scatter(df_[df_.stroke == 0].bmi, df_[df_.stroke == 0].age, s = 40, marker = 'o'
            , linewidths = 1, edgecolors = 'lightblue', facecolors = 'white', alpha = 0.6
            , label = 'Did Have Stroke')

ax1.set_ylim(ymin =0)
ax1.set_ylabel('BMI')
ax1.set_xlim(xmin = 0)
ax1.set_xlabel('Age')
ax1.legend(loc = 'upper left', frameon = False, fontsize = 'x-small')
plt.show()
c_palette = { 'No': 'lightblue', 'Yes': 'orange'}
sns.boxplot( x = 'hypertension', y = 'age', data= dfc, orient= 'v', ax = ax2, palette = c_palette)
sns.boxplot('hypertension', 'bmi', data=dfc, orient = 'v', ax = ax3, plaette = c_palette)
gs.tight_layout(plt.gcf())
