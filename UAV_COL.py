
# define function for waypoints generation
# inputs - design variables
# output - waypoints for the two UAVs


# define function for trajectory generation
# slightly complicated


# define function for controls
# slightly complicated


# define function for dynamics
# slightly complicated 

import numpy as np
import math 
from Trajectory_generator import trajectory_generation2 

def waypoints_generation(actions, input_params, other_params):
	t2=actions[0]
	delta_v = actions[1]
	theta = actions[2]
	t1=0.0

	Pa = np.array(input_params[0])
	Va = np.array(input_params[1])
	Pb = np.array(input_params[2])
	Vb = np.array(input_params[3])
	#print(Va[0])

	t3=other_params[0]
	t4 = t3+(t3-t2)
	t5=other_params[1]
	heading_A = np.arctan(Va[1]/Va[0])
	heading_B = np.arctan(Vb[1]/Vb[0])
	heading_new_A = (heading_A + theta)%(2.0*np.pi)
	heading_new_B = (heading_B + theta)%(2.0*np.pi)

	speed_A = math.sqrt(Va[0]**2 + Va[1]**2)
	speed_B = math.sqrt(Vb[0]**2 + Vb[1]**2)

	if speed_A > speed_B:
		speed_new_A = speed_A + delta_v
		speed_new_B = speed_B - delta_v
	else:
		speed_new_A = speed_A - delta_v
		speed_new_B = speed_B + delta_v
		
	P1a = Pa
	P1b = Pb
	
	P2a = P1a + np.multiply(Va,(t2-t1))
	P2b = P1b + np.multiply(Vb,(t2-t1))


	
	V_new_A = np.multiply([math.cos(heading_new_A), math.sin(heading_new_A)],speed_new_A)
	V_new_B = np.multiply([math.cos(heading_new_B), math.sin(heading_new_B)],speed_new_B)
	
	P3a = P2a + np.multiply(V_new_A,(t3-t2))
	P3b = P2b + np.multiply(V_new_B,(t3-t2))
	
	P4a = P1a + np.multiply(Va,(t4-t1))
	P4b = P1b + np.multiply(Vb,(t4-t1))

	V_new2_A = np.multiply((P4a - P3a),1.0/(t4-t3))
	V_new2_B = np.multiply((P4b - P3b),1.0/(t4-t3))
	#print(P4a)
	P5a = P1a + np.multiply(Va,(t5-t1))
	P5b = P1b + np.multiply(Vb,(t5-t1))
	WP_A = ([P1a,P2a,P3a,P4a,P5a])
	WP_B = [P1b,P2b,P3b,P4b,P5b]
	T = [t1,t2,t3,t4,t5]
	return WP_A, WP_B, T 	

class traj_inputs(object):
	V1 = [0.0, 0.0]
	tm = 3.0
	waypoints = []
	time_int = []
	length=6
	def __init__(self, V1, tm, waypoints, time_int):
		self.V1 = V1
		self.tm = tm
		self.waypoints = waypoints
		self.time_int = time_int
		self.length  = 6


def trajectory_generation(inputs):
	if inputs.length == 6:
		vel = inputs.V1
		tmax = inputs.tm
		waypoints = (inputs.waypoints)
		n1 = np.size(waypoints,0)
		n2 = np.size(waypoints,1)
		n=n1-1
		#print(waypoints[0][0])
		#print(n,n1,n2)
		traj_time=inputs.time_int
		S= traj_time
		A=np.zeros((8*n,8*n))
		b=np.zeros((8*n,1))
		#print(S[0])
		for i in range(n):
			A[i][(i)*8] = 1.0
			#print(waypoints[0][i])
			b[i] = waypoints[i][0]
		# n constraints

		for i in range(n,2*n):
			t1 = i%n
			for j in range(t1*8,t1*8+8):
				A[i][j]=1.0
			b[i]=waypoints[t1+1][0]	
		A[2*n][1]=1/(S[1]-S[0])
		b[2*n][0]=vel[0]
		# 2n+1 constraints

		for i in range(1,8):
			A[2*n+1][(n-1)*8+i]=(i)/(S[n]-S[n-1])
		b[2*n+1]=vel[0]
		# 2n+2 constraints


		cn=0
		for i in range(2*n+2,3*n+1):
			cn=cn+1
			pw=2
			for j in range((cn-1)*8+1,(cn-1)*8+8):
				A[i][j]=(pw-1)/(S[cn]-S[cn-1])
				pw=pw+1
			j=cn*8 +1
			A[i][j] = -1.0/(S[cn+1]-S[cn])
			b[i]=0.0	


		cn=0
		for i in range(3*n+1,4*n):
			cn=cn+1
			pw=3
			for j in range((cn-1)*8+2,(cn-1)*8+8):
				A[i][j]=(pw-1)*(pw-2)/((S[cn]-S[cn-1])**2)
				pw=pw+1;
			j=cn*8 +2
			A[i][j] = -(2.0/(S[cn+1]-S[cn])**2)
			b[i]=0.0
		# 4n constraints done

		cn=0
		for i in range(4*n,5*n-1):
			cn=cn+1
			pw=4
			for j in range((cn-1)*8+3,(cn-1)*8+8):
				A[i][j]=(pw-1)*(pw-2)*(pw-3)/((S[cn]-S[cn-1])**3)
				pw=pw+1;
			j=cn*8 +3
			A[i][j] = -(6.0/(S[cn+1]-S[cn])**3)
			b[i]=0.0
		# n constrain

		cn=0
		for i in range(5*n-1,6*n-2):
			cn=cn+1
			pw=5
			for j in range((cn-1)*8+4,(cn-1)*8+8):
				A[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)/((S[cn]-S[cn-1])**4)
				pw=pw+1;
			j=cn*8 +4
			A[i][j] = -(24.0/(S[cn+1]-S[cn])**4)
			b[i]=0.0


		cn=0
		for i in range(6*n-2,7*n-3):
			cn=cn+1
			pw=6
			for j in range((cn-1)*8+5,(cn-1)*8+8):
				A[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)*(pw-5)/((S[cn]-S[cn-1])**5)
				pw=pw+1;
			j=cn*8 +5
			A[i][j] = -(120.0/(S[cn+1]-S[cn])**5)
			b[i]=0.0	

		cn=0	
		for i in range(7*n-3,8*n-4):
			cn=cn+1
			pw=7
			for j in range((cn-1)*8+6,(cn-1)*8+8):
				A[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)*(pw-5)*(pw-6)/((S[cn]-S[cn-1])**6)
				pw=pw+1;
			j=cn*8 +6
			A[i][j] = -(720.0/(S[cn+1]-S[cn])**6)
			b[i]=0.0		
		# 8n-4 constraints done
		A[8*n-4][2]=2.0/((S[1]-S[0])**2)
		b[8*n-4]=0.0
		
		for i in range(3,8):
			A[8*n-3][(n-2)*8+i-1]=(i-1)*(i-2)/((S[n]-S[n-1])**2)
		
		b[8*n-3]=0.0
		
		A[8*n-2][3]=6.0/((S[1]-S[0])**3)
		b[8*n-2]=0.0

		for i in range (4,8):
			A[8*n-1][(n-2)*8+i-1]=(i-1)*(i-2)*(i-3)/((S[n]-S[n-1])**3)
		
		b[8*n-1]=0.0
					
		return A, b
		



	#if len(inputs) == 2:	


#def controller(t, state, des_state, params):
#	if t==0:





other_params=np.array([5.0,10.0])
input_params=([[0.0,0.0],[1.0,1.0],[2.0,0.0],[-1.0,1.0]])
action=np.array([3.2,0.0,0.0])
WP_A, WP_B, T = waypoints_generation(action, input_params, other_params)
#print(input_params)
#print(np.array(WP_A[2]))
#gt=(np.array(WP_A[2])+ np.array(WP_A[1]))
#print(WP_A[0][0])
inputs = traj_inputs(input_params[1], 10.0, WP_A, T)
A, b, A2, b2 = trajectory_generation2(inputs)
A_inv = np.linalg.inv(A)
alpha = np.matmul(A_inv,b)
print(np.size(alpha,1))
print(alpha)
#print(waypoints_generation(action, input_params, other_params))

