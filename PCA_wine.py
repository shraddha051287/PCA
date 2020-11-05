import pandas as pd 
import numpy as np
wine = pd.read_csv("C:\\Users\\hp\\Downloads\\wine.csv")
wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
wine.data = wine
wine.head(4)

# Normalizing the numerical data 
wine_normal = scale(wine)

pca = PCA(n_components = 6)
pca_values = pca.fit_transform(wine_normal)


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]
pca.components_[1]
# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:2:3]
plt.scatter(x,y,color=["red","blue"])

from mpl_toolkits.mplot3d import Axes3D
Axes3D.scatter(np.array(x),np.array(y),np.array(z),c=["green","blue","red"])


################### Clustering  ##########################
new_df = pd.DataFrame(pca_values[:,0:4])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)


###### screw plot or elbow curve ###########

TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(3):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(3,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(new_df)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
Univ['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]

Univ.iloc[:,1:7].groupby(Univ.clust).mean()

Univ.to_csv("Univsersity.csv")
