## Script for calculating the negative log likelihood
from sklearn.neighbors.kde import KernelDensity
import numpy as np

#function to choose sigma*
def get_sigma(x_sample, x_valid, sigmas=None):
	if sigmas is None:
		sigmas=np.logspace(-1.,0.,10)
	ll_valid=[]
	for sigma in sigmas:
		P_x_fake=KernelDensity(kernel='gaussian',bandwidth=sigma).fit(x_sample)
		ll_valid.append(P_x_fake.score_samples(x_valid).mean())
		print sigma, ll_valid[-1]
	return sigmas[np.argmax(ll_valid)]	

def ll(x_sample, x_valid, x_test,sigmas=None):
	sigma=get_sigma(x_sample, x_valid, sigmas)
	P_x_fake=KernelDensity(kernel='gaussian', bandwidth=sigma).fit(x_sample)
	return P_x_fake.score_samples(x_test).mean(), sigma

