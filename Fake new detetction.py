import numpy as np
import pandas as  pd
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split

#Reading the data into python and seeing few lines of the data to understand the dimensions... 
df=pd.read_csv('C:\\Users\\DELL\\Downloads\\work\\Projects data flair\\news.csv',encoding='latin')
print(df.head())
print(df.shape)
labels=df.label
print(labels.head())

#splitting the data into training and testing data...
X_train,X_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2,random_state=6)

#Initializing TfidfVectorizer and removing the stop words
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)

# Fitting and transforming the train set, transforming the test set
tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)

#Now time to train
model=PassiveAggressiveClassifier(max_iter=50,random_state=6)
model.fit(tfidf_train,y_train)

# time to predict
y_pred=model.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(score*100)
print(confusion_matrix(y_test,y_pred,labels=['FAKE','REAL']))

