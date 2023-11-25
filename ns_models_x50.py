# importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')# data visualisation and manipulationimport numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import missingno as msno#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
# %matplotlib inline
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)#import the necessary modelling algos.
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#model selection
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score
from sklearn.model_selection import GridSearchCV#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler

urlx='drive/MyDrive/NS/Lab/MachineLearningCSV/CSVs/X50_prepro.csv'
urly='drive/MyDrive/NS/Lab/MachineLearningCSV/CSVs/y_prepro.csv'
x = pd.read_csv(urlx)
y =pd.read_csv(urly)

x.shape

y.shape

x1=x.drop(['Unnamed: 0'],axis=1)

y1=y.drop(['Unnamed: 0'],axis=1)

x1.shape

y1.shape

# let's us split the data and target into training and testing
# import train_test_split library from sklearn's model_selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.33, random_state=42)

"""#### running models

##### KNN
"""

#importing libraries for logistic regression
from sklearn.neighbors import KNeighborsClassifier
mlKN1= KNeighborsClassifier(n_neighbors=4, weights='distance',algorithm='ball_tree',p=2)
import time
stime = time.time()
mlKN1.fit(x_train,y_train)
print("--- %s seconds needed to train ---" % (time.time() - stime))

y_pred=mlKN1.predict(x_test)
y_pred=pd.DataFrame(y_pred.reshape(-1,1))
y_pred.head(4)

y_pred_KN1_n=pd.DataFrame(y_pred_KN1.reshape(-1,1))
# let us check accuracy
# import metrics library
from sklearn import metrics
print("accuracy is: ", metrics.accuracy_score(y_test,y_pred_KN1_n))

# import confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred_KN1_n))
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("f1 score: ", f1_score(y_test,y_pred_KN1_n,average='weighted'))
print("precision score: ", precision_score(y_test, y_pred_KN1_n, average='weighted'))
print("recall score: ", recall_score(y_test, y_pred_KN1_n, average='weighted'))



"""#### random forest"""

# let us create and run different models on the dataset
# let us use randomforest
# import libraries
from sklearn.ensemble import RandomForestClassifier
ml=RandomForestClassifier(n_estimators=100)
import time
stime = time.time()
ml.fit(x_train,y_train)
print("--- %s seconds needed to train ---" % (time.time() - stime))

y_pred=ml.predict(x_test)
y_pred=pd.DataFrame(y_pred.reshape(-1,1))
y_pred.head(4)

# let us check accuracy
# import metrics library
from sklearn import metrics
print("accuracy is: ", metrics.accuracy_score(y_test,y_pred))
# import confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("f1 score: ", f1_score(y_test,y_pred,average='weighted'))
print("precision score: ", precision_score(y_test, y_pred, average='weighted'))
print("recall score: ", recall_score(y_test, y_pred, average='weighted'))

"""#### SVM"""

# let us create and run different models on the dataset
# let us use randomforest
# import libraries
from sklearn.svm import SVC
ml1=SVC()
import time
stime = time.time()
ml1.fit(x_train,y_train)
print("--- %s seconds needed to train ---" % (time.time() - stime))

y_predS=ml1.predict(x_test)
y_predS=pd.DataFrame(y_predS.reshape(-1,1))
y_predS.head(4)

# let us check accuracy
# import metrics library
from sklearn import metrics
print("accuracy is: ", metrics.accuracy_score(y_test,y_predS))
# import confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_predS))
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("f1 score: ", f1_score(y_test,y_predS,average='weighted'))
print("precision score: ", precision_score(y_test, y_predS, average='weighted'))
print("recall score: ", recall_score(y_test, y_predS, average='weighted'))



"""#### Adaboost"""

# let us create and run different models on the dataset
# let us use randomforest
# import libraries
from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier(random_state=96)
import time
stime = time.time()
clf.fit(x_train,y_train)
print("--- %s seconds needed to train ---" % (time.time() - stime))
print("training score: ",clf.score(x_train, y_train))

y_pred1=clf.predict(x_test)
y_pred1=pd.DataFrame(y_pred1.reshape(-1,1))
y_pred1.head(4)

# let us check accuracy
# import metrics library
from sklearn import metrics
print("accuracy is: ", metrics.accuracy_score(y_test,y_pred1))
# import confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred1))
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("f1 score: ", f1_score(y_test,y_pred1,average='weighted'))
print("precision score: ", precision_score(y_test, y_pred1, average='weighted'))
print("recall score: ", recall_score(y_test, y_pred1, average='weighted'))

"""#### decision tree"""

# let us create and run different models on the dataset
# let us use randomforest
# import libraries
from sklearn import tree
clf=tree.DecisionTreeClassifier(criterion="gini")
import time
stime = time.time()
clf.fit(x_train,y_train)
print("--- %s seconds needed to train ---" % (time.time() - stime))
print("training score: ",clf.score(x_train, y_train))

y_pred2=clf.predict(x_test)
y_pred2=pd.DataFrame(y_pred2.reshape(-1,1))
y_pred2.head(4)

# let us check accuracy
# import metrics library
from sklearn import metrics
print("accuracy is: ", metrics.accuracy_score(y_test,y_pred2))
# import confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred2))
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("f1 score: ", f1_score(y_test,y_pred2,average='weighted'))
print("precision score: ", precision_score(y_test, y_pred2, average='weighted'))
print("recall score: ", recall_score(y_test, y_pred2, average='weighted'))









