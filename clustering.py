import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
    

def sol(D,C,phi):
    kmeans=KMeans(n_clusters=C,init=phi,algorithm='full').fit(D)
    centers=pd.DataFrame(kmeans.cluster_centers_,columns={'x','y'})
    centers['labels']=[k for k in range(C)]
    labels=kmeans.labels_
    df=pd.DataFrame(data=D,columns={'x','y'})
    df['labels']=labels
    print(centers)
    print(means)

    sns.relplot(data=df,x='x',y='y',hue='labels',palette="colorblind")
    sns.scatterplot(data=centers,x='x',y='y',color="black")
    
    return None

if __name__=="__main__":
    np.random.seed(123)
    rng=np.random.default_rng

    means=[np.array([0,0]),np.array([10,0]),np.array([0,6]),np.array([9,8])]
    cov=[np.identity(2),np.matrix([[1,0.2],[0.2,1.5]]),
            np.matrix([[1,0.4],[0.4,1.1]]),np.matrix([[0.3,0.2],[0.2,0.3]])]

    D0=rng().multivariate_normal(means[0],cov[0],250)
    D1=rng().multivariate_normal(means[1],cov[1],250)
    D2=rng().multivariate_normal(means[2],cov[2],250)
    D3=rng().multivariate_normal(means[3],cov[3],250)
    D=np.concatenate((D0,D1,D2,D3))
    np.random.default_rng().shuffle(D)

    phi=np.random.default_rng().random(size=(5,2,))

    #Teil 1
    sol(D,4,phi[:4])
    #Teil 2
    sol(D,3,phi[:3])
    #Teil 3
    sol(D,5,phi)
    #Teil 4
    phi=np.array([[-2,-2],[-2.1,-2.1],[-2,-2.2],[-2.1,-2.2]])
    sol(D,4,phi)
    #Teil 5
    phi[3]=np.array([30,30])
    sol(D,4,phi)

    #show plots
    plt.show()