import pyKriging  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import csv
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.ensemble import RandomForestRegressor 

input = open('input_2.csv','r')
output = open('output_2.csv', 'r')

sl=0
#print(input)
X=np.zeros((2000,5))
Y =np.zeros(2000)
for row in csv.reader(input):
	
	#X[sl] = int(row)
	X[sl] = list(map(float, row))
	#print(X[sl])
	sl +=1
	
sl=0
for rw in csv.reader(output):
	#print(row)
	Y[sl] = float(rw[0])
	sl +=1		
#print(X,Y)
input.close()
output.close()
#print(max(Y))
gp = RandomForestRegressor(n_estimators=5)
mod = gp.fit(X[0:1951], Y[0:1951])
print(sum(abs(mod.predict(X[1951:2001])- Y[1951:2001])))	
#testfun = pyKriging.testfunctions().branin
#k = kriging(X[0:10], Y[0:10], testfunction=testfun, name='simple', testPoints=250)  
#k.train()
#k.snapshot()

