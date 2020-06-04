import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import datetime

df=pd.read_excel('E:/DS/DATA/psi.xlsx')
df.isna().sum()
df['timestamp']=df['timestamp'].agg(pd.Timestamp)
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.to_period('d')
df1 = df.groupby(['timestamp']).mean()
df1['mean']=df1.iloc[:,0:6].mean(axis=1)
df1['lable']=np.zeros(len(df1['mean']))
df1['lable'].loc[df1['mean']<=55]='1'
df1['lable'].loc[df1['mean'].between(55,151)]='2'
df1['lable'].loc[df1['mean'].between(150,251)]='3'
df1['lable'].loc[df1['mean']>=251]='4'

X=df1.iloc[:,0:7]
Y=Y=df1.iloc[:,7]

aclog=np.empty(1000)
a#clda=np.empty(1000)
a#cqda=np.empty(1000)
acgnb=np.empty(1000)
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #lda = LinearDiscriminantAnalysis()
    #lda.fit(X_train, y_train)
    #qda = QuadraticDiscriminantAnalysis()
    #qda.fit(X_train, y_train)
    logreg = LogisticRegression(multi_class="multinomial",solver="newton-cg")
    logreg.fit(X_train, y_train)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    aclog[i]=logreg.score(X_test,y_test)
    #aclda[i]=lda.score(X_test,y_test)
    #acqda[i]=qda.score(X_test,y_test)
    acgnb[i]=gnb.score(X_test,y_test)

print('prediction accuracies for Logreg,GNB are \n {},{}'.format(np.mean(aclog),np.mean(acgnb)))
    
