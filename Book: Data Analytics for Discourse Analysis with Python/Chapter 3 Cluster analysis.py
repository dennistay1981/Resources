
"""
Chapter 3 Cluster analysis
"""
#import Python libraries
import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np  
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist

#import dataset
data=pd.read_csv('covid.csv',index_col='Country') 

#scale data to zero mean and unit variance
scaler=StandardScaler()   
data=pd.DataFrame(scaler.fit_transform(data),columns=data.columns,index=data.index)


"""
AHC
"""
#generate dendrogram
plt.title("COVID-19 dendrogram",fontsize=25)  
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('Distance',fontsize=25)
plt.xlabel('Country/region',fontsize=25)
dend = dendrogram(linkage(data, metric='euclidean',method='ward'),orientation='top', leaf_rotation=80,leaf_font_size=25,labels=data.index,color_threshold=10)  
plt.show()


#calculate cophenetic correlation coefficient
c, coph_dist = cophenet(shc.linkage(data,method='ward',metric='euclidean'), pdist(data))
c



"""
K-means clustering
"""
#import dataset and scale scores
data=pd.read_csv('A.csv',index_col='Session')  
scaler=StandardScaler()
scaler.fit(data) 
data=pd.DataFrame(scaler.transform(data),columns=data.columns,index=data.index)



#determine the optimal number of clusters with the 'elbow method'
from sklearn.cluster import KMeans
n=8   #n can be changed to test more clusters
num_clusters = range(1, n+1)
inertias = []

for i in num_clusters:
    model=KMeans(n_clusters=i)
    model.fit(data)
    inertias.append(model.inertia_)
    
#generate 'elbow plot'
plt.plot(num_clusters, inertias, '-o')
plt.xlabel('number of clusters, k',fontsize=15)
plt.ylabel('inertia value',fontsize=15)
plt.title('Dyad A (CBT)',fontsize=15)
plt.xticks(num_clusters, fontsize=15)
plt.show()

#confirm visual inspection with actual inertia value change
pd.DataFrame(inertias).diff().abs()


#generate cluster centroids and labels using the optimal number of clusters
model = KMeans(n_clusters=3)   #the value of n_clusters should be set to the optimal k
labels = model.fit_predict(data)
data['cluster_labels'] = labels

#obtain cluster centroid positions for later plotting
cluster_centres = model.cluster_centers_




"""
Generate 2D-scatterplot to visualize clustering solution
"""
from sklearn.decomposition import PCA as sklearnPCA
#specify two principal components
pca = sklearnPCA(n_components=2)
#reduce the cluster centroid locations into two dimensions
cent=pca.fit_transform(cluster_centres).T 
#use data.iloc to remove cluster labels in the rightmost column before reducing the data
reduced=pd.DataFrame(pca.fit_transform(data.iloc[:,:-1]),columns=['Dim_1','Dim_2'],index=data.index)  
#reattach previous cluster labels to prepare for plotting
reduced['cluster']=data['cluster_labels']  
  
#generate scatterplot and color according to clusters
sns.scatterplot(x='Dim_1', y='Dim_2', hue='cluster', data=reduced, palette='tab10', s=30)

#plot cluster centroids
plt.plot(cent[0],cent[1],'rx',markersize=15)

#annotate each object
for i in range(reduced.shape[0]):
    plt.text(x=reduced.Dim_1[i]+0.05, y=reduced.Dim_2[i]+0.05, s=reduced.index[i],
             fontdict=dict(color='black',size=10))   
plt.legend(title='cluster')
plt.xlabel("Dimension 1",fontsize=15)
plt.ylabel("Dimension 2",fontsize=15)
plt.title("Dyad A (Psychoanalysis)",fontsize=15)
plt.show()


"""
Model validation
"""
#visual validation by reconstructing cluster centroids 
data.groupby('cluster_labels').mean().plot(kind='bar')
plt.show()

#check and plot cluster sizes
data.groupby('cluster_labels').count()
sns.countplot(data=data,x=data['cluster_labels'])

#validation with logistic regression 
#import Python libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#split dataset into features and target variable
X = data.iloc[:,:-1] 
y = data['cluster_labels'] 

#instantiate the model. multi_class='auto' detects if outcome is binary or multinomial
logreg = LogisticRegression(multi_class='auto')
logreg.fit(X,y)
#get percentage accuracy
logreg.score(X,y) 

#generate confusion matrix
cnf_matrix = confusion_matrix(logreg.predict(X), y)   
cnf_matrix






