"""
Chapter 4 Classification
"""
#import Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 

#import dataset and scale scores
data=pd.read_csv('data.csv',index_col='Type')
scaler=StandardScaler()   
data=pd.DataFrame(scaler.fit_transform(data),columns=data.columns,index=data.index)


"""
Initial visualization of pre-existing groups
"""
#reduce dataset to 2D and plot distribution of pre-existing groups
from sklearn.decomposition import PCA as sklearnPCA
pca = sklearnPCA(n_components=2)
data2D=pd.DataFrame(pca.fit_transform(data),columns=['Dim_1','Dim_2'],index=data.index)  
sns.scatterplot(data=data2D,x='Dim_1',y='Dim_2',hue='Type')


"""
The k_NN process
"""
#create arrays for the predictors and outcome
X=data.values
y=data.index.values


#generate random training set with stratify=y to ensure fair split among types
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0, stratify=y)


"""
Determine the optimal value of k 
"""
#setup arrays to store various predictive accuracies
neighbors = np.arange(1, 21)
test_accuracy = np.empty(len(neighbors))
train_accuracy = np.empty(len(neighbors))
overall_accuracy = np.empty(len(neighbors))

#loop over values of k, fitting a k-NN model and computing accuracies each time
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)  
    #compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test) 
    #compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #compute accuracy on the whole set 
    overall_accuracy[i] = knn.score(X, y)
    

#generate plot. note that the plot illustrates bias-variance tradeoff
plt.title('Accuracy with different k')
plt.plot(neighbors, test_accuracy, label = 'Testing accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training accuracy')
plt.plot(neighbors, overall_accuracy, label = 'Overall accuracy')
plt.plot(neighbors,(test_accuracy+train_accuracy+overall_accuracy)/3, ':',label = 'Average')
plt.legend()
plt.xticks(neighbors)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


#(re)fit the model to the training data, specifying optimal k
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)


"""
Model validation
"""
#(re)print accuracy measures (test, train, overall)
knn.score(X_test, y_test)  
knn.score(X_train, y_train) 
knn.score(X, y)  

#predict labels for the testing set
test_pred = knn.predict(X_test)

#generate confusion matrix
metrics.confusion_matrix(y_test, test_pred, labels=["CBT","HUM","PA"])

#generate classification report
print(metrics.classification_report(y_test, test_pred))

#k-folds cross validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(knn,X, y,cv=10, scoring='accuracy')  #can change scoring parameter to 'precision', 'recall', 'f1' etc. (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
print(cv_scores)
print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))









