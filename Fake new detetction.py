import numpy as np
import pandas as  pd
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss,precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


#To ignore depreciation warnings we use this function
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

#Reading the data into python and seeing few lines of the data to understand the dimensions... 
df=pd.read_csv('C:\\Users\\DELL\\Downloads\\work\\Projects data flair\\news.csv',encoding='latin')
print(df.columns)
df['label']=df['label'].replace(['REAL','FAKE'],[0,1])
print(df.label)
labels=df.label


#splitting the data into training and testing data...
X_train,X_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2,random_state=6)

#Initializing TfidfVectorizer and removing the stop words
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)

# Fitting and transforming the train set, transforming the test set
tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)

#Storing all the classfiers to be tested in a list
classifiers = [
    KNeighborsClassifier(3),
    XGBClassifier(objective="binary:logistic",random_state=7),
    
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    
    PassiveAggressiveClassifier(max_iter=50,random_state=6)]


# Making 2 lists to store the accuracy scores & f1 scores of each classifier to compare 
accs=[]
f1s=[]


for clf in classifiers:
    clf.fit(tfidf_train,y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    y_pred=clf.predict(tfidf_test)
    acc = accuracy_score(y_test, y_pred)
    accs.append(acc)
    print("Accuracy: {:.4%}".format(acc))
    f1 = f1_score(y_test, y_pred)
    f1s.append(f1)
    print("F1 score: {:.4%}".format(f1))
    
   
    
print("="*30)

print(f1s)

#Plotting the scores obtained from each model to have a better idea of performance
labels=['K-neighbors','XGB','Decision Tree','Ranndom Forest','Adaboost','Grad Boost','Passive Aggresive']
x=range(len(f1s))
plt.plot( f1s,color='red',marker='s' )
plt.xticks(x, labels, rotation ='vertical')
plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()




#Output Snippet
# Index(['Unnamed: 0', 'title', 'text', 'label'], dtype='object')
# 0       1
# 1       1
# 2       0
# 3       1
# 4       0
#        ..
# 6330    0
# 6331    1
# 6332    1
# 6333    0
# 6334    0
# Name: label, Length: 6335, dtype: int64
# ==============================
# KNeighborsClassifier
# ****Results****
# Accuracy: 56.9850%
# F1 score: 69.3648%
# [19:32:12] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
# ==============================
# XGBClassifier
# ****Results****
# Accuracy: 93.7648%
# F1 score: 93.6546%
# ==============================
# DecisionTreeClassifier
# ****Results****
# Accuracy: 83.7411%
# F1 score: 83.2792%
# ==============================
# RandomForestClassifier
# ****Results****
# Accuracy: 91.4759%
# F1 score: 91.2338%
# ==============================
# AdaBoostClassifier
# ****Results****
# Accuracy: 87.2139%
# F1 score: 86.9775%
# ==============================
# GradientBoostingClassifier
# ****Results****
# Accuracy: 90.7656%
# F1 score: 90.6921%
# ==============================
# PassiveAggressiveClassifier
# ****Results****
# Accuracy: 93.5280%
# F1 score: 93.4609%
# ==============================
# [0.6936481169196179, 0.9365461847389558, 0.8327922077922079, 0.9123376623376622, 0.8697749196141479, 0.9069212410501193, 0.9346092503987242]
 
