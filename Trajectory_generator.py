import numpy as np
import math 

def trajectory_generation_coef(inputs):
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

		# for y

		A2=np.zeros((8*n,8*n))
		b2=np.zeros((8*n,1))
		#print(S[0])
		for i in range(n):
			A2[i][(i)*8] = 1.0
			#print(waypoints[0][i])
			b2[i] = waypoints[i][1]
		# n constraints

		for i in range(n,2*n):
			t1 = i%n
			for j in range(t1*8,t1*8+8):
				A2[i][j]=1.0
			b2[i]=waypoints[t1+1][1]	
		A2[2*n][1]=1/(S[1]-S[0])
		b2[2*n][0]=vel[1]
		# 2n+1 constraints

		for i in range(1,8):
			A2[2*n+1][(n-1)*8+i]=(i)/(S[n]-S[n-1])
		b2[2*n+1]=vel[1]
		# 2n+2 constraints


		cn=0
		for i in range(2*n+2,3*n+1):
			cn=cn+1
			pw=2
			for j in range((cn-1)*8+1,(cn-1)*8+8):
				A2[i][j]=(pw-1)/(S[cn]-S[cn-1])
				pw=pw+1
			j=cn*8 +1
			A2[i][j] = -1.0/(S[cn+1]-S[cn])
			b2[i]=0.0	


		cn=0
		for i in range(3*n+1,4*n):
			cn=cn+1
			pw=3
			for j in range((cn-1)*8+2,(cn-1)*8+8):
				A2[i][j]=(pw-1)*(pw-2)/((S[cn]-S[cn-1])**2)
				pw=pw+1;
			j=cn*8 +2
			A2[i][j] = -(2.0/(S[cn+1]-S[cn])**2)
			b2[i]=0.0
		# 4n constraints done

		cn=0
		for i in range(4*n,5*n-1):
			cn=cn+1
			pw=4
			for j in range((cn-1)*8+3,(cn-1)*8+8):
				A2[i][j]=(pw-1)*(pw-2)*(pw-3)/((S[cn]-S[cn-1])**3)
				pw=pw+1;
			j=cn*8 +3
			A2[i][j] = -(6.0/(S[cn+1]-S[cn])**3)
			b2[i]=0.0
		# n constrain

		cn=0
		for i in range(5*n-1,6*n-2):
			cn=cn+1
			pw=5
			for j in range((cn-1)*8+4,(cn-1)*8+8):
				A2[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)/((S[cn]-S[cn-1])**4)
				pw=pw+1;
			j=cn*8 +4
			A2[i][j] = -(24.0/(S[cn+1]-S[cn])**4)
			b2[i]=0.0


		cn=0
		for i in range(6*n-2,7*n-3):
			cn=cn+1
			pw=6
			for j in range((cn-1)*8+5,(cn-1)*8+8):
				A2[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)*(pw-5)/((S[cn]-S[cn-1])**5)
				pw=pw+1;
			j=cn*8 +5
			A2[i][j] = -(120.0/(S[cn+1]-S[cn])**5)
			b2[i]=0.0	

		cn=0	
		for i in range(7*n-3,8*n-4):
			cn=cn+1
			pw=7
			for j in range((cn-1)*8+6,(cn-1)*8+8):
				A2[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)*(pw-5)*(pw-6)/((S[cn]-S[cn-1])**6)
				pw=pw+1;
			j=cn*8 +6
			A2[i][j] = -(720.0/(S[cn+1]-S[cn])**6)
			b2[i]=0.0		
		# 8n-4 constraints done
		A2[8*n-4][2]=2.0/((S[1]-S[0])**2)
		b2[8*n-4]=0.0
		
		for i in range(3,8):
			A2[8*n-3][(n-2)*8+i-1]=(i-1)*(i-2)/((S[n]-S[n-1])**2)
		
		b2[8*n-3]=0.0
		
		A2[8*n-2][3]=6.0/((S[1]-S[0])**3)
		b2[8*n-2]=0.0

		for i in range (4,8):
			A2[8*n-1][(n-2)*8+i-1]=(i-1)*(i-2)*(i-3)/((S[n]-S[n-1])**3)
		
		b2[8*n-1]=0.0




		# for Z

		A3=np.zeros((8*n,8*n))
		b3=np.zeros((8*n,1))
		#print(S[0])
		for i in range(n):
			A3[i][(i)*8] = 1.0
			#print(waypoints[0][i])
			b3[i] = waypoints[i][2]
		# n constraints

		for i in range(n,2*n):
			t1 = i%n
			for j in range(t1*8,t1*8+8):
				A3[i][j]=1.0
			b3[i]=waypoints[t1+1][2]	
		A3[2*n][1]=1/(S[1]-S[0])
		b3[2*n][0]=vel[2]
		# 2n+1 constraints

		for i in range(1,8):
			A3[2*n+1][(n-1)*8+i]=(i)/(S[n]-S[n-1])
		b3[2*n+1]=vel[2]
		# 2n+2 constraints


		cn=0
		for i in range(2*n+2,3*n+1):
			cn=cn+1
			pw=2
			for j in range((cn-1)*8+1,(cn-1)*8+8):
				A3[i][j]=(pw-1)/(S[cn]-S[cn-1])
				pw=pw+1
			j=cn*8 +1
			A3[i][j] = -1.0/(S[cn+1]-S[cn])
			b3[i]=0.0	


		cn=0
		for i in range(3*n+1,4*n):
			cn=cn+1
			pw=3
			for j in range((cn-1)*8+2,(cn-1)*8+8):
				A3[i][j]=(pw-1)*(pw-2)/((S[cn]-S[cn-1])**2)
				pw=pw+1;
			j=cn*8 +2
			A3[i][j] = -(2.0/(S[cn+1]-S[cn])**2)
			b3[i]=0.0
		# 4n constraints done

		cn=0
		for i in range(4*n,5*n-1):
			cn=cn+1
			pw=4
			for j in range((cn-1)*8+3,(cn-1)*8+8):
				A3[i][j]=(pw-1)*(pw-2)*(pw-3)/((S[cn]-S[cn-1])**3)
				pw=pw+1;
			j=cn*8 +3
			A3[i][j] = -(6.0/(S[cn+1]-S[cn])**3)
			b3[i]=0.0
		# n constrain

		cn=0
		for i in range(5*n-1,6*n-2):
			cn=cn+1
			pw=5
			for j in range((cn-1)*8+4,(cn-1)*8+8):
				A3[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)/((S[cn]-S[cn-1])**4)
				pw=pw+1;
			j=cn*8 +4
			A3[i][j] = -(24.0/(S[cn+1]-S[cn])**4)
			b3[i]=0.0


		cn=0
		for i in range(6*n-2,7*n-3):
			cn=cn+1
			pw=6
			for j in range((cn-1)*8+5,(cn-1)*8+8):
				A3[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)*(pw-5)/((S[cn]-S[cn-1])**5)
				pw=pw+1;
			j=cn*8 +5
			A3[i][j] = -(120.0/(S[cn+1]-S[cn])**5)
			b3[i]=0.0	

		cn=0	
		for i in range(7*n-3,8*n-4):
			cn=cn+1
			pw=7
			for j in range((cn-1)*8+6,(cn-1)*8+8):
				A3[i][j]=(pw-1)*(pw-2)*(pw-3)*(pw-4)*(pw-5)*(pw-6)/((S[cn]-S[cn-1])**6)
				pw=pw+1;
			j=cn*8 +6
			A3[i][j] = -(720.0/(S[cn+1]-S[cn])**6)
			b3[i]=0.0		
		# 8n-4 constraints done
		A3[8*n-4][2]=2.0/((S[1]-S[0])**2)
		b3[8*n-4]=0.0
		
		for i in range(3,8):
			A3[8*n-3][(n-2)*8+i-1]=(i-1)*(i-2)/((S[n]-S[n-1])**2)
		
		b3[8*n-3]=0.0
		
		A3[8*n-2][3]=6.0/((S[1]-S[0])**3)
		b3[8*n-2]=0.0

		for i in range (4,8):
			A3[8*n-1][(n-2)*8+i-1]=(i-1)*(i-2)*(i-3)/((S[n]-S[n-1])**3)
		
		b3[8*n-1]=0.0

		A_inv = np.linalg.inv(A)
		A2_inv = np.linalg.inv(A2)
		A3_inv = np.linalg.inv(A3)
		alpha = np.matmul(A_inv,b)
		alpha2 = np.matmul(A2_inv,b2)
		alpha3 = np.matmul(A3_inv,b3)

					
		return alpha, alpha2, alpha3

class desired_state(object):

	def __init__(self):
		self.pos = [0.0, 0.0, 0.0]
		self.vel = [0.0, 0.0, 0.0]
		self.acc = [0.0, 0.0, 0.0]
		self.yaw = 0.0
		self.yawdot = 0.0

	def state_def(self,pos,vel,yaw,yawdot):	
		self.pos = pos
		self.vel = vel
		self.yaw = yaw
		self.yawdot = yawdot


def trajectory_desired_state(t,S,alpha,alpha2,alpha3):
	#print(S)
	n= len(S)-1
	des_state = desired_state()
	for i in range(n):
		if t>= S[i] and t<=S[i+1]:
			sm=0.0
			pw=0
			sm2=0.0
			sm3=0.0
			#des_state = desired_state()

			for j in range((i)*8,(i)*8+8):
				sm=sm+ alpha[j]*((t-S[i])/(S[i+1]-S[i]))**pw
				sm2=sm2+ alpha2[j]*((t-S[i])/(S[i+1]-S[i]))**pw
				sm3=sm3+ alpha3[j]*((t-S[i])/(S[i+1]-S[i]))**pw
				pw=pw+1

			des_state.pos = np.array([sm,sm2,sm3])

			sm=0.0
			pw=0
			sm2=0.0
			sm3=0.0
			for j in range((i)*8+1,(i)*8+8):
				sm=sm+ (alpha[j]*((t-S[i])/(S[i+1]-S[i]))**pw)/(S[i+1]-S[i])*(pw+1)
				sm2=sm2+ (alpha2[j]*((t-S[i])/(S[i+1]-S[i]))**pw)/(S[i+1]-S[i])*(pw+1)
				sm3=sm3+ (alpha3[j]*((t-S[i])/(S[i+1]-S[i]))**pw)/(S[i+1]-S[i])*(pw+1)
				pw=pw+1
			des_state.vel = np.array([sm,sm2,sm3])	


			sm=0.0
			pw=0
			sm2=0.0
			sm3=0.0
			for j in range((i)*8+2,(i)*8+8):
				sm=sm+ (alpha[j]*((t-S[i])/(S[i+1]-S[i]))**pw)/((S[i+1]-S[i])**2)*((pw+1)*(pw+2))
				sm2=sm2+ (alpha2[j]*((t-S[i])/(S[i+1]-S[i]))**pw)/((S[i+1]-S[i])**2)*((pw+1)*(pw+2))
				sm3=sm3+ (alpha3[j]*((t-S[i])/(S[i+1]-S[i]))**pw)/((S[i+1]-S[i])**2)*((pw+1)*(pw+2))
				pw=pw+1
			des_state.acc = np.array([sm,sm2,sm3])	

			#des_state.yaw = math.atan2(des_state.vel[1],des_state.vel[0])
			des_state.yaw = (math.atan2(des_state.vel[1],des_state.vel[0]) + 2*np.pi)%(2*np.pi)
			des_state.yawdot = math.atan2(des_state.acc[1],des_state.acc[0])
	return des_state		


