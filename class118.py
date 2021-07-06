from numpy.random import RandomState
import plotly.express as px
import pandas as pd
import csv

df = pd.read_csv("./petals_sepals_data.csv")
fig = px.scatter(df,x = "petal_size", y = "sepal_size")
fig.show()

# -----------------------------------------------------------
# To find the possible value of K using the WCSS perimeter

from sklearn.cluster import KMeans

X = df.iloc[:,[0,1]].values

#dataframe.iloc[] method is used to retrive the rows when the index of the dataframe is something other than numeric series(0,1,2,3,4,5,6,7,8,9)
#index is like an address that's how any data point across the dataframe can be accessed 
#rows and columns both have indexes 
#Rows ndices are called as index
#For columns it's the general column names 

print(X)
wcss = []
#we just need 10 cluster points 

for i in range (1,11):
    kmeans = KMeans(n_clusters=i,init = 'k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#In the classifier we are passing the number of clusters we want to use for classifier(i)
#RandomState requeired to tell the classifier from where it should start 
#It fits our list we created finding the inertia of our data

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='p', color='green')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#We have observed that in the elbow chart that the wcss value is decreasing until the K = 3
#Hence we can see that K = 3 for the data given
#From the scattered plot earlier wem got that their might be 3 clusters

#Putting the number of clusters as 3

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1')
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids',s=100,marker=',')
plt.grid(False)
plt.title('Clusters of Flowers')
plt.xlabel('Petal Size')
plt.ylabel('Sepal Size')
plt.legend()
plt.show()