from scipy.integrate import odeint
import random
import numpy as np
import time as TIME


def tester_function(s,t, a, b, c):
	dt = np.zeros(13)
	for i in range(13):
		
		dt[i]= np.cos(t)*a*b*c
	return dt

x0 = np.zeros(13)
timeint = np.linspace(0.0, np.pi, 6)
start = TIME.time()
a=10.0
b=9.3
c=10.0
xsave = odeint(tester_function, x0, timeint, args=(a,b,c), full_output=1, rtol=0.0000000001)
print(TIME.time()-start)
#print(xsave[0][5][1])

	