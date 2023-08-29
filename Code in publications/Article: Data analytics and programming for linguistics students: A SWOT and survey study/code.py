
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('data.csv',index_col='ID')


# Figure 1
fig,axs=plt.subplots(2,2,sharex=False)
plt.setp(axs, yticks=np.arange(0,6,0.5)) 
plt.tight_layout(pad=2)
sns.pointplot(x="Category",y="ABILITY",data = data,ax=axs[0,0], join=False)
sns.stripplot(x="Category",y="ABILITY",color='black',alpha=0.3, data=data,ax=axs[0,0])
sns.pointplot(x="Category",y="INTEREST",data = data,ax=axs[0,1], join=False)
sns.stripplot(x="Category",y="INTEREST",color='black',alpha=0.3,data=data,ax=axs[0,1])
sns.pointplot(x="Category",y="P_GROWTH",data = data,ax=axs[1,0], join=False)
sns.stripplot(x="Category",y="P_GROWTH",color='black',alpha=0.3,data=data,ax=axs[1,0])
sns.pointplot(x="Category",y="UTILITY",data = data,ax=axs[1,1], join=False)
sns.stripplot(x="Category",y="UTILITY",color='black',alpha=0.3,data=data,ax=axs[1,1])
fig.set_size_inches(12, 8)
fig.text(0.5, 0.05, 'Mean scores across learner dispositions', ha='center', va='center',fontsize=14) 


# Figure 2 
data2 = data[['ABILITY','INTEREST','P_GROWTH','UTILITY']]
data2['Category']= data['Category']

sns.pairplot(data=data2, hue='Category', diag_kind='rug')



# Create arrays for the features and the response variable
y = data['Category']

x = data[['ABILITY','INTEREST','P_GROWTH','UTILITY']]
y1 = data['SW']
y2 = data['OT']



"""
Nested CV using knn
"""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()

# y1: Predict S vs. W
param_grid =  {'n_neighbors': [2, 3, 4, 5, 6, 7, 8], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)
inner_cv = KFold(n_splits=2, shuffle=True, random_state=123)

grid_search = GridSearchCV(knn, param_grid=param_grid, cv=inner_cv)
nested_score = cross_val_score(grid_search, X=x, y=y1, cv=outer_cv, scoring='accuracy')

print("Nested CV score: %.3f +/- %.3f" % (nested_score.mean(), nested_score.std()))


data['SW_pred']=cross_val_predict(grid_search, x, y1)
accuracy = accuracy_score(data['SW'], data['SW_pred'])
cnf_matrix = confusion_matrix(data['SW'], data['SW_pred']) 


sns.heatmap(cnf_matrix, annot=True, cmap="Blues", yticklabels=data.groupby('SW').count().index, 
            xticklabels=data.groupby('SW').count().index,  annot_kws={"size": 18},
            linewidths=1.5, linecolor='black')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Predicting S vs. W')




# y2: Predict O vs. T
grid_search = GridSearchCV(knn, param_grid=param_grid, cv=inner_cv)
nested_score = cross_val_score(grid_search, X=x, y=y2, cv=outer_cv, scoring='accuracy')

print("Nested CV score: %.3f +/- %.3f" % (nested_score.mean(), nested_score.std()))

data['OT_pred']=cross_val_predict(grid_search, x, y2)
accuracy = accuracy_score(data['OT'], data['OT_pred'])

cnf_matrix = confusion_matrix(data['OT'], data['OT_pred'])  #real, then predicted

sns.heatmap(cnf_matrix, annot=True, cmap="Blues", yticklabels=data.groupby('OT').count().index, 
            xticklabels=data.groupby('OT').count().index,  annot_kws={"size": 18}, 
            linewidths=1.5, linecolor='black')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Predicting O vs. T')



# Generate final predictions by joining SW_pred with OT_pred
data['final_pred']=data['SW_pred']+data['OT_pred']

print(classification_report(data['Category'], data['final_pred']))
accuracy_score(data['Category'], data['final_pred'])



cnf_matrix = confusion_matrix(data['Category'], data['final_pred'])  

sns.heatmap(cnf_matrix, annot=True, cmap="Blues", yticklabels=data.groupby('Category').count().index, 
            xticklabels=data.groupby('Category').count().index,  annot_kws={"size": 18})
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

























































"""
Ensemble learning
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# INSTANTIATE 3 DIFF CLASSIFIERS (Logreg, knn, decision tree)
# Instantiate lr
lr = LogisticRegression(random_state=1)
# Instantiate knn
knn = KNN(n_neighbors=3)
# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.4, random_state=1)   #each leaf contains at least 13% of training data
# Define the list of classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:    
    # Fit clf to the training set
    clf.fit(X_train, y_train)    
    # Predict y_pred
    y_pred = clf.predict(X_test) 
    # Calculate accuracy
    accuracy = accuracy_score(y_pred, y_test) 
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))
  
    
#USE VOTING CLASSIFIER
# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier
# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)     
# Fit vc to the training set
vc.fit(X_train, y_train)   
# Evaluate the test set predictions
y_pred = vc.predict(X_test)
# Calculate accuracy score
accuracy = accuracy_score(y_test,y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))




#Visualize VOTING CLASSIFIER outcomes
#Confusion matrix with raw frequencies and percentages
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)  #real, then predicted
cnf_matrix
#Heatmap with actual numbers (rows=actual labels, columns=predicted labels)
data.groupby('Category').count()

sns.heatmap(cnf_matrix, annot=True, cmap="Blues", yticklabels=['SO','ST','WO','WT'],xticklabels=['SO','ST','WO','WT'],  annot_kws={"size": 15})
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

























