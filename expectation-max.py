import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.mixture import GaussianMixture

def generate_data():
    rng=np.random.default_rng

    mu1=np.array([1,1])
    mu2=np.array([3,3])
    mu3=np.array([2,6])
    means=[mu1,mu2,mu3]

    c1=0.1*np.identity(2)
    c2=0.2*np.identity(2)
    c3=0.3*np.identity(2)
    c=[c1,c2,c3]

    data=[]

    for _ in range(600):
        k=np.random.randint(0,5)
        if k in {0,1}:
            data.append(rng().multivariate_normal(means[0],c[0]))
        elif k in {2,3}:
            data.append(rng().multivariate_normal(means[1],c[1]))
        else:
            data.append(rng().multivariate_normal(means[2],c[2]))
    data=np.asarray(data)
    data=np.reshape(data,(600,2))
    return data

def estimate(Dat,parameters,GMM):
    GMM.set_params(**parameters)
    est=GMM.fit_predict(Dat)
    return est


if __name__=="__main__":
    Dat=generate_data()
    
    D=pd.DataFrame(Dat,columns=['x1','x2'])
    sns.jointplot(data=D,x='x1',y='x2')
    GMM=GaussianMixture()
    parameters_dict={
        0:{"n_components":3,
            "covariance_type":"spherical"},
        1:{"n_components":3,
            "covariance_type":"spherical",
            "means_init":[[0,2],[5,2],[5,5]],
            "precisions_init":[1/0.15,1/0.27,1/0.4],
            "weights_init":[1/3]*3},
        2:{"n_components":2,
            "covariance_type":"spherical",
            "means_init":[[1.6,1.4],[1.4,1.6]],
            "precisions_init":[1/0.2,1/0.4],
            "weights_init":[0.5]*2}
        }
   
    for i in range(3):
        D["y"+str(i)]=estimate(Dat,parameters_dict[i],GMM)
        print(i,":\n means:",GMM.means_,"\n\n Covariance",GMM.covariances_,"\n")
        sns.jointplot(data=D,x='x1',y='x2',hue='y'+str(i),palette="colorblind")
    
    plt.show()
