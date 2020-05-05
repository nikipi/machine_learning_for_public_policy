import time, json, requests
import pandas as pd
import numpy as np
from collections import Counter
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
import seaborn as sns
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)



url = r'https://raw.githubusercontent.com/canghailan/Wuhan-2019-nCoV/master/Wuhan-2019-nCoV.csv'
data = pd.read_csv(url, sep=',',engine='python')


data=data[data['provinceCode'] == 420000]

print(data['city'].value_counts())

data['date']=pd.to_datetime(data['date'])
data=data.set_index('date')


df=data.truncate(after='2020-03-30',before='2020-01-25')

df=df.loc[:,['city','cityCode','confirmed','suspected','cured','dead']]
df=df.dropna()


city_dict = Counter(df['city'])
city_list = [key for key in city_dict.keys()]


date_dict = Counter(df.index.values)
date_list = [str(key) for key in date_dict.keys()]
print(date_list.sort)

target_list = ['confirmed', 'suspected', 'cured', 'dead']
target_df = pd.DataFrame({'city':city_list})
for key in date_list:
    sub_df = df.loc[key]
    new_column_name = [key[:10] + '_' + target for target in target_list]
    confirmed = []
    suspected = []
    cured = []
    dead = []
    for city in city_list:
        city_info = sub_df[sub_df['city']==city]
        if len(city_info) == 0:
            confirmed.append('nan')
            suspected.append('nan')
            cured.append('nan')
            dead.append('nan')
        else:
            confirmed.append(int(city_info['confirmed']))
            suspected.append(int(city_info['suspected']))
            cured.append(int(city_info['cured']))
            dead.append(int(city_info['dead']))
    date_df = pd.DataFrame({'city':city_list, new_column_name[0]:confirmed, new_column_name[1]:suspected, new_column_name[2]:cured, new_column_name[3]:dead})
    target_df = pd.merge(target_df, date_df, on='city')

target_df.to_csv('/Users/apple/Desktop/wuhan_data.csv',encoding='utf_8_sig')

df2=target_df.set_index('city')
df2 = df2.filter(regex='cured')

df2.to_csv('/Users/apple/Desktop/wuhancured_data.csv',encoding='utf_8_sig')



plt.rcParams['font.sans-serif']=['Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False


data['dead_rate']=(data.cumdead/data.cumconfirmed)*100
data['cured_rate']=(data.cumcured/data.cumconfirmed)*100



print(data.columns)
X=data[['facilities／1k', 'bed', 'staff', 'doctor', 'nurse']]
y=data['cured_rate']



'''
#train data distribution
plt.scatter(X.bed, y,  color='blue')
plt.xlabel("bed")
plt.ylabel("cured_raten")
plt.show()

from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)
encoded = lab_enc.fit_transform(X)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

mreg=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
mreg.fit(x_train,y_train)
pred_test= mreg.predict(x_test)

# calculating mse

mse = np.mean((pred_test - y_test)**2)
print('mse is :',mse)

# evaluation using r-square
print('square is:',mreg.score(x_test,y_test))

coef = mreg.fit(X, y).coef_
print(coef)

# Plot the coefficients
plt.plot(range(len(X.columns)), coef)

plt.margins(0.02)
plt.show()



steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets


# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline,param_grid=parameters)

# Fit to the training set
cv.fit(X_train,y_train)
'''

import numpy as np
from sklearn import linear_model
from sklearn import svm

X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=21)
classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]

trainingData    = X_train
trainingScores  = y_train
predictionData  = X_test

for item in classifiers:
    print(item)
    clf = item
    clf.fit(trainingData, trainingScores)
    print(clf.predict(predictionData),'\n')
    pred_test = clf.predict(predictionData)
    mse = np.mean((pred_test - y_test) ** 2)
    print('mse is :', mse)

    # evaluation using r-square
    print('square is:', clf.score(X_test, y_test))


'''
clf = LinearRegression()
clf.fit(trainingData, trainingScores)
print("LinearRegression")
print(clf.predict(predictionData))

clf = svm.SVR()
clf.fit(trainingData, trainingScores)
print("SVR")
print(clf.predict(predictionData))

clf = LogisticRegression()
clf.fit(trainingData, trainingScores)
print("LogisticRegression")
print(clf.predict(predictionData))

clf = DecisionTreeClassifier()
clf.fit(trainingData, trainingScores)
print("DecisionTreeClassifier")
print(clf.predict(predictionData))

clf = KNeighborsClassifier()
clf.fit(trainingData, trainingScores)
print("KNeighborsClassifier")
print(clf.predict(predictionData))

clf = LinearDiscriminantAnalysis()
clf.fit(trainingData, trainingScores)
print("LinearDiscriminantAnalysis")
print(clf.predict(predictionData))

clf = GaussianNB()
clf.fit(trainingData, trainingScores)
print("GaussianNB")
print(clf.predict(predictionData))

clf = SVC()
clf.fit(trainingData, trainingScores)
print("SVC")
print(clf.predict(predictionData




LogisticRegression is not for regression but classification !

The Y variable must be the classification class,

(for example 0 or 1)

And not a continuous variable,

that would be a regression problem.

shareimprove this answerfollow


'''

