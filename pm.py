import numpy as np
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
df1['lable'].loc[df1['mean']<=55]='1'#normal
df1['lable'].loc[df1['mean'].between(55,151)]='2'#elevated
df1['lable'].loc[df1['mean'].between(150,251)]='3'#high
df1['lable'].loc[df1['mean']>=251]='4'#very high

X=df1.iloc[:,0:7]
Y=df1.iloc[:,7]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

logreg = LogisticRegression(multi_class="multinomial",solver="newton-cg")
logreg.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)



print('LDA Model for Training Data\n',confusion_matrix(y_train,lda.predict(X_train)))
print('LDA Model for Testing Data\n',confusion_matrix(y_test,lda.predict(X_test)))
print('Total prediction accuracy for Training Data\n',lda.score(X_train,y_train))
print('Total prediction accuracy for Testing Data\n',lda.score(X_test,y_test))

print('QDA Model for Training Data\n',confusion_matrix(y_train,qda.predict(X_train)))
print('QDA Model for Testing Data\n',confusion_matrix(y_test,qda.predict(X_test)))
print('Total prediction accuracy for Training Data\n',qda.score(X_train,y_train))
print('Total prediction accuracy for Testing Data\n',qda.score(X_test,y_test))

print('Logistic Regression Model for Training Data\n',confusion_matrix(y_train,logreg.predict(X_train)))
print('Logistic Regression Model for Testing Data\n',confusion_matrix(y_test,logreg.predict(X_test)))
print('Total prediction accuracy for Training Data\n',logreg.score(X_train,y_train))
print('Total prediction accuracy for Testing Data\n',logreg.score(X_test,y_test))


print('Naive Bayes Model for Training Data\n',confusion_matrix(y_train,gnb.predict(X_train)))
print('Naive Bayes Model for Testing Data\n',confusion_matrix(y_test,gnb.predict(X_test)))
print('Total prediction accuracy for Training Data\n',gnb.score(X_train,y_train))
print('Total prediction accuracy for Testing Data\n',gnb.score(X_test,y_test))

print('The prediction accuracies for Logistic regression, LDA, QDA and Naive Bayes methods are \n {},{},{} and {}, respectively'.format(logreg.score(X_test,y_test),lda.score(X_test,y_test),qda.score(X_test,y_test),gnb.score(X_test,y_test)))

