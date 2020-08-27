import pandas as pd

from sklearn.cluster import MeanShift

if __name__=='__main__':
    dataset=pd.read_csv('./data/candy.csv')
    print(dataset.head)

    X= dataset.drop('competitorname',axis=1)

    menashift= MeanShift().fit(X)
    print(max(menashift.labels_))
    print('='*64)
    print(menashift.cluster_centers_)

    dataset['meanshift']=menashift.labels_
    print(dataset)