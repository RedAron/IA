#clousterisacion 
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__=="__main__":
    dataset=pd.read_csv('./data/candy.csv')
    
    X= dataset.drop('competitorname',axis=1)

    kmeans= MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("total de centros:", len(kmeans.cluster_centers_))

    print("="*64)
    print(kmeans.predict(X))

    dataset['groups']= kmeans.predict(X)

    print(dataset)


