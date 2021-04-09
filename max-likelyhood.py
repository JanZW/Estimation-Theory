import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#Problem 1

#Part 1

def create_multivariat(mean, cov, n,show):
	"""create_multivariat returns np.ndarray of 2 feature vectors of a 
	Gaussian multivariate distribution.
	mean: 1-D array-like of length N; determines the mean of the distribution
	cov: 2-D array-like of shape (N,N); determines the covariance of the 
		distribution
	n: int; number of created samples
	show: bool; if true: create jointplot of created samples"""
	if n==1:
		x=np.random.default_rng().multivariate_normal(mean, cov)
	else:
		x=np.random.default_rng().multivariate_normal(mean, cov, n)
	if show:
	 	df=pd.DataFrame({'x':x[:,0],'y':x[:,1]})
	 	sns.jointplot(data=df,x='x',y='y')
	return x

'''def create_multivariat2(means,k,cov):
	x=np.random.default_rng().multivariate_normal(means[k-1],cov)
	return x'''

def multivariat_ml(x):
	"""multivarait_ml calculates a maximum likelihood estimate of the provided
	input, assuming a Gaussian multivariat distribution.
	x: np.ndarray; mulitvariat Gaussian data
	"""
	numel, dim=x.shape
	mu=x.mean(axis=0)
	cov=np.zeros((dim,dim))
	for dt in x:
		cov+=1/numel*np.outer((dt-mu),(dt-mu))
	return mu, cov

def estimate(means,arr):
	arr=arr[0]
	distances=np.array([np.linalg.norm(arr-m) for m in means])
	return np.argmin(distances)+1

def part1(mean,cov,n):
	"""part1: Solve part 1 of exercise 1 on sheet I: Estimation Theory
	mean: array-like;
	cov: 2D array;
	n: int"""
	x= create_multivariat(mean, cov, n, True)
	est_mean, est_cov=multivariat_ml(x) 
	return est_mean,est_cov

#Part 2
def part2(cov,means):
	"""part2: Solves part 2 of exercise 1 on sheet I: Estimation Theory
	cov: 2-D array"""
	
	D1={1:[],2:[],3:[]}
	D2=[]
	for _ in range(1000):
		k=np.random.randint(1,4)
		l=np.random.randint(1,4)
		D1[k].append(create_multivariat(means[k],cov,1,False))
		D2.append({"x":create_multivariat(means[l],cov,1,False),"k":l})
	est_means={}
	est_cov={}
	for m in range(1,4):
		D1[m]=np.asarray(D1[m])
		est_means[m],est_cov[m]=multivariat_ml(D1[m])
	D2=pd.DataFrame(D2)
	est_cov=(est_cov[1]+est_cov[2]+est_cov[3])/3
	
	D2['e']=D2.apply(lambda row: estimate(est_means,row),axis=1)
	accuracy=len(D2.query('k==e'))/len(D2)
	return est_means,est_cov,accuracy

if __name__=="__main__":
	print('\n------Problem 1: Part 1------------------\n')
	mean=np.array([2,-2])
	cov=np.array([[0.9,0.2],[0.2,0.3]])
	est_mean,est_cov=part1(mean,cov,50)
	est_mean2,est_cov2=part1(mean,cov,5000)

	print('estimated mean:\n', est_mean, '\n estimated cov:\n', est_cov)
	print('actual mean:\n', mean, '\n actual cov:\n', cov)
	print('error mean:\n', mean-est_mean,'\n error cov:\n', cov-est_cov)
	print("mean norm:\n",np.linalg.norm(mean-est_mean))
	print("cov norm:\n",np.linalg.norm(cov-est_cov))
	print('created 5000 samples')
	print('estimated mean:\n', est_mean2,'\n estimated cov:\n', est_cov2)
	print('error mean:\n', mean-est_mean2,'\n error cov:\n', cov-est_cov)
	print("mean norm:\n",np.linalg.norm(mean-est_mean2))
	print("cov norm:\n",np.linalg.norm(cov-est_cov2))
	plt.show()

	print('\n------Problem 1: Part 2------------------\n')
	cov=np.array([[0.8,0,0],[0,0.8,0],[0,0,0.8]])
	means={1:np.array([0,0,0]),2:np.array([1,2,2]),3:np.array([3,3,4])}
	est_means,est_cov,accuracy=part2(cov,means)
	print('estimated mean:\n', est_means, '\n estimated cov:\n', est_cov)
	print('actual means:\n', means, '\n actual cov:\n', cov)
	error_means=[]
	error_cov=[]
	for k in range(1,4):
		error_means.append(means[k]-est_means[k])
	error_cov.append(cov-est_cov)
	print('error mean:\n', error_means,'\n error cov:\n', error_cov)
	print("accuracy",accuracy)

	cov=np.array([[0.8,0.2,0.1],[0.2,0.8,0.2],[0.1,0.2,0.8]])
	est_means,est_cov,accuracy=part2(cov,means)
	print('estimated means:\n', est_means, '\n estimated cov:\n', est_cov)
	print('actual means:\n', means, '\n actual cov:\n', cov)
	print("accuracy",accuracy)