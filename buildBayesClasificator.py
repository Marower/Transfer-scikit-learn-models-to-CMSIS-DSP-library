from sklearn.naive_bayes import GaussianNB
from loadDataSet import getDataset
from saveClassifiers import saveBayesClasificator;
import numpy as np
import time

X_train, X_test, y_train, y_test = getDataset()

gnb = GaussianNB()
gnb.fit(X_train, y_train)
start = time.time()
for i in range(10):
    y_pred = gnb.predict(X_test)
end = time.time()
print(end - start)

print("Number of mislabeled points out of a total %d points : %d", (len(X_test), (y_test != y_pred).sum()))

# Dump of data for CMSIS-DSP

print("Parameters")
# Gaussian averages
theta = list(np.reshape(gnb.theta_, np.size(gnb.theta_)))
print("Theta = ", theta)

# Gaussian variances
Sigma = list(np.reshape(gnb.var_, np.size(gnb.var_)))
print("Sigma = ", Sigma)

# Class priors
classPriors = list(np.reshape(gnb.class_prior_, np.size(gnb.class_prior_)))
print("Prior = ", classPriors)

print("Epsilon = ", gnb.epsilon_)

saveBayesClasificator(gnb)
