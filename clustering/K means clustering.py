import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,3:].values



'''from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()'''




kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)



plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],c='red')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],c='blue')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],c='green')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],c='cyan')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],c='magenta')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',s=300)
plt.xlabel=('annual income')
plt.ylabel('spending score')
plt.show()
