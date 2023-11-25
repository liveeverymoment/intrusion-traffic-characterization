# importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing



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

#importing the dataset
url="drive/MyDrive/NS/Lab/MachineLearningCSV/CSVs/dataset.csv"
df = pd.read_csv(url)

print(df.shape)
df1=df.drop(['Unnamed: 0'], axis =1)
N=1000000
df2=df1.groupby('Label', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(df1))))).sample(frac=1).reset_index(drop=True)

df1["Label"].value_counts()

df2

df2.shape

df2.groupby(['Label']).transform('count').nunique()

# Replacing infinite with nan
df2.replace([np.inf, -np.inf], np.nan, inplace=True)

# Dropping all the rows with nan values
df2.dropna(inplace=True)

# Printing df
df2

df2=df2.drop(['Destination_Port'],axis=1)

df2.shape

df2["Label"].value_counts()

df2.isnull().values.all()

df2

X = df2.drop(['Label'], axis=1)
X.head()

for col in X:
  print("col name: ",col," no of unique classes: ",X[col].nunique())

X.shape

y = df2['Label']
y.head()
print("label no of unique classes: ", y.nunique())

y.shape

y

# Integer Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)

y1= pd.DataFrame(y)

y1.shape

y1

# sns.countplot(data=y1)
# y1.value_counts()

# one hot encoding for y
# Integer Encoding
label_encoder = LabelEncoder()
y_label = label_encoder.fit_transform(y)
print(y_label)


#reshaping for OneHotEncoder
y_label_reshape = y_label.reshape(len(y_label), 1)

# One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse=False)
y_one = onehot_encoder.fit_transform(y_label_reshape)
print(y_one)

# Normalizing numerical columns
# define standard scaler
scaler = preprocessing.MinMaxScaler()
# transform data
for col in X:
  print("normalized feature using Minmax scaler: ",col)
  X[col]=scaler.fit_transform(pd.to_numeric(X[col],errors='coerce').values.reshape(-1,1))

#pd.DataFrame(y_one, columns=['predictions']).to_csv('y_one.csv')

X.shape

#saving dataframe to csv in the drive
X.to_csv(r'drive/MyDrive/NS/Lab/MachineLearningCSV/CSVs/X_prepro.csv')

X50 = X[['Average_Packet_Size','Packet_Length_Std', 'Packet_Length_Mean', 'Packet_Length_Variance', 'Subflow_Bwd_Bytes','Total_Length_of_Bwd_Packets','Avg_Bwd_Segment_Size','Bwd_Packet_Length_Mean','Init_Win_bytes_forward','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Bwd_Packet_Length_Max','Max_Packet_Length','Init_Win_bytes_backward','Fwd_Packet_Length_Max','Flow_IAT_Max','Flow_Bytess','Flow_Duration','Fwd_IAT_Max','Fwd_Header_Length','Fwd_Header_Length.1','Avg_Fwd_Segment_Size','Fwd_Packet_Length_Mean','Fwd_Packetss','Bwd_Packetss','Fwd_IAT_Total','Flow_Packetss','Bwd_Header_Length','Flow_IAT_Mean','Fwd_IAT_Mean','Bwd_Packet_Length_Std','Flow_IAT_Std','Fwd_Packet_Length_Std','Fwd_IAT_Std','Bwd_IAT_Max','Subflow_Bwd_Packets','Total_Backward_Packets','Bwd_IAT_Total','Subflow_Fwd_Packets','Total_Fwd_Packets','Bwd_IAT_Mean','Bwd_Packet_Length_Min','Idle_Max','min_seg_size_forward','Idle_Mean','Idle_Min','Bwd_IAT_Std','Fwd_IAT_Min','Active_Mean']]

X40 = X[['Average_Packet_Size','Packet_Length_Std', 'Packet_Length_Mean', 'Packet_Length_Variance', 'Subflow_Bwd_Bytes','Total_Length_of_Bwd_Packets','Avg_Bwd_Segment_Size','Bwd_Packet_Length_Mean','Init_Win_bytes_forward','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Bwd_Packet_Length_Max','Max_Packet_Length','Init_Win_bytes_backward','Fwd_Packet_Length_Max','Flow_IAT_Max','Flow_Bytess','Flow_Duration','Fwd_IAT_Max','Fwd_Header_Length','Fwd_Header_Length.1','Avg_Fwd_Segment_Size','Fwd_Packet_Length_Mean','Fwd_Packetss','Bwd_Packetss','Fwd_IAT_Total','Flow_Packetss','Bwd_Header_Length','Flow_IAT_Mean','Fwd_IAT_Mean','Bwd_Packet_Length_Std','Flow_IAT_Std','Fwd_Packet_Length_Std','Fwd_IAT_Std','Bwd_IAT_Max','Subflow_Bwd_Packets','Total_Backward_Packets','Bwd_IAT_Total','Subflow_Fwd_Packets']]

X50

X40

#saving dataframe to csv in the drive
X40.to_csv(r'drive/MyDrive/NS/Lab/MachineLearningCSV/CSVs/X40_prepro.csv')

#saving dataframe to csv in the drive
X50.to_csv(r'drive/MyDrive/NS/Lab/MachineLearningCSV/CSVs/X50_prepro.csv')

print(y1.index.name)

y1.head()

y1.rename(columns={0:'Label'}, inplace=True)

y1.nunique()

y1.shape

y1.to_csv(r'drive/MyDrive/NS/Lab/MachineLearningCSV/CSVs/y_prepro.csv')

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=101)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

X

x_train.head(5)

"""#### running random forest and decision tree"""

models=[RandomForestClassifier(), DecisionTreeClassifier()]
model_names=['RandomForestClassifier','DecisionTree']
acc=[]
d={}
for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    acc.append(accuracy_score(pred,y_test))

d={'Modelling Algo':model_names,'Accuracy':acc}

acc_frame=pd.DataFrame(d)
acc_frame

param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv= 5)
CV_rfc.fit(x_train, y_train)

print("Best score : ",CV_rfc.best_score_)
print("Best Parameters : ",CV_rfc.best_params_)
print("Precision Score : ", precision_score(CV_rfc.predict(x_test),y_test))

df1 = pd.DataFrame.from_records(x_train)
tmp = pd.DataFrame({'Feature': df1.columns, 'Feature importance': clf_rf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()



"""#### running models
random forest
"""

# let us create and run different models on the dataset
# let us use randomforest
# import libraries
from sklearn.ensemble import RandomForestClassifier
ml=RandomForestClassifier(n_estimators=100)
import time
stime = time.time()
ml.fit(X_train,y_train)
print("--- %s seconds needed to train ---" % (time.time() - stime))
y_pred=ml.predict(X_test)
y_pred=pd.DataFrame(y_pred.reshape(-1,1))
y_pred.head(4)

# from sklearn.preprocessing import LabelEncoder
# le= LabelEncoder()
# y_test=le.fit_transform(y_test)
# y_test

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

"""##### Ada boost algorithm"""

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
y_pred=clf.predict(x_test)
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

"""#### stacking ensemble classifier"""

X

y1

# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models

# define dataset
#X, y = data[:, :-1], data[:, -1]
X, y = X, y1
# get the models to evaluate
models = get_models()

# get the dataset
# def get_dataset():
# 	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# 	return X, y

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()



"""#### voting ensemble classifier"""

# compare hard voting to standalone classifiers
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
	return X, y

# get a voting ensemble of models
def get_voting():
	# define the base models
	models = list()
	models.append(('knn1', KNeighborsClassifier(n_neighbors=1)))
	models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
	models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
	models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
	models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# get a list of models to evaluate
def get_models():
	models = dict()
	models['knn1'] = KNeighborsClassifier(n_neighbors=1)
	models['knn3'] = KNeighborsClassifier(n_neighbors=3)
	models['knn5'] = KNeighborsClassifier(n_neighbors=5)
	models['knn7'] = KNeighborsClassifier(n_neighbors=7)
	models['knn9'] = KNeighborsClassifier(n_neighbors=9)
	models['hard_voting'] = get_voting()
	return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()



