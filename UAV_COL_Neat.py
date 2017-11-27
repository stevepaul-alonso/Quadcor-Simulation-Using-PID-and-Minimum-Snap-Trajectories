import os
import neat
from pyDOE import *
import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow
import math 
from scipy.integrate import odeint
from Trajectory_generator import trajectory_generation_coef, trajectory_desired_state, desired_state
import time as TIME
import csv
#import pyKriging
#from pyKriging.krige import kriging  
#from pyKriging.samplingplan import samplingplan
plt.axis([-60, 60, -60, 60])
plt.ion()

# calculates the waypoints WP_A, WP_B, and time instant 'T'
def waypoints_generation(actions, input_params, other_params):
	t2=actions[0]
	delta_v = actions[1]
	theta = actions[2]
	t1=0.0

	Pa = np.array(input_params[0])
	Va = np.array(input_params[1])
	Pb = np.array(input_params[2])
	Vb = np.array(input_params[3])

	t3=other_params[0]
	t4 = t3+(t3-t2)
	t5=other_params[1]
	heading_A = (math.atan2(Va[1],Va[0]) + 4.0*np.pi)%(2.0*np.pi)
	heading_B = (math.atan2(Vb[1],Vb[0]) + 4.0*np.pi)%(2.0*np.pi)
	heading_new_A = (heading_A + theta + 4.0*np.pi)%(2.0*np.pi)
	heading_new_B = (heading_B + theta + 4.0*np.pi)%(2.0*np.pi)
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
	V_new_A = np.multiply([math.cos(heading_new_A), math.sin(heading_new_A), 0.0],speed_new_A)
	V_new_B = np.multiply([math.cos(heading_new_B), math.sin(heading_new_B), 0.0],speed_new_B)
	
	P3a = P2a + np.multiply(V_new_A,(t3-t2))
	P3b = P2b + np.multiply(V_new_B,(t3-t2))
	
	P4a = P1a + np.multiply(Va,(t4-t1))
	P4b = P1b + np.multiply(Vb,(t4-t1))

	V_new2_A = np.multiply((P4a - P3a),1.0/(t4-t3))
	V_new2_B = np.multiply((P4b - P3b),1.0/(t4-t3))
	P5a = P1a + np.multiply(Va,(t5-t1))
	P5b = P1b + np.multiply(Vb,(t5-t1))
	WP_A = [P1a,P2a,P3a,P4a,P5a]
	WP_B = [P1b,P2b,P3b,P4b,P5b]
	T = [t1,t2,t3,t4,t5]
	Time_int = np.linspace(t1,t5,51)
	waypoints_A = np.zeros((51,3))
	dt = (t5-t1)/50
	Po = Pa
	for i in range(51):
		waypoints_A[i] = Po
		if Time_int[i]<=t2:
			V = Va
		elif Time_int[i]>t2 and Time_int[i]<=t3:
			V = V_new_A
		elif Time_int[i]>t3 and Time_int[i]<=t4:
			V = V_new2_A
		else:
			V = Va
		Po = Po + np.multiply(V,dt)

	waypoints_B = np.zeros((51,3))
	dt = (t5-t1)/50
	Po2 = Pb
	for i in range(51):
		waypoints_B[i] = Po2
		if Time_int[i]<=t2:
			V = Vb
		elif Time_int[i]>t2 and Time_int[i]<=t3:
			V = V_new_B
		elif Time_int[i]>t3 and Time_int[i]<=t4:
			V = V_new2_B
		else:
			V = Vb
		Po2 = Po2 + np.multiply(V,dt)
					
	return waypoints_A, waypoints_B, Time_int	

class system_parameters(object):
	def __init__(self):
		self.m = 1.0
		self.g = 9.81
		self.mass = self.m
		self.I = [[0.00025,0.0, .00000255],[0,0.000232,0.0],[.00000255,0.0,0.0003738]]
		self.invI = np.linalg.inv(self.I)
		self.gravity = 9.81
		self.arm_length = 0.086
		self.minF = 0.0
		self.maxF = 20.0*self.m*self.g
			


class traj_inputs(object):
	V1 = [0.0, 0.0, 0.0]
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

# initialise the state variables 's'
def init_state(start,vel,yaw): # state initialise
	s = np.zeros(17)
	phi0 = 0.0
	theta0 = 0.0
	psi0 = yaw
	Rot0 = RPY2Rot_ZXY(phi0,theta0,psi0)
	Quat0 = Rot2Quat(Rot0)
	s[0] = start[0]
	s[1] = start[1]
	s[2] = start[2]
	s[3] = vel[0]
	s[4] = vel[1]
	s[5] = vel[2]
	s[6] = Quat0[0]
	s[7] = Quat0[1]
	s[8] = Quat0[2]
	s[9] = Quat0[3]
	s[10] = 0.0
	s[11] = 0.0
	s[12] = 0.0
	s[13] = 0.0
	s[14] = 0.0
	s[15] = 0.0
	s[16] = 0.0

	return s

# calculates the quaternion given the rotation matrix R
def Rot2Quat(R):
	tr = R[0][0] + R[1][1] + R[2][2]

	if tr>0.0:
		S = (np.sqrt(tr + 1.0))*2.0
		qw = 0.25*S
		qx = (R[2][1] - R[1][2])/S
		qy = (R[0][2] - R[2][0])/S
		qz = (R[1][0] - R[0][1])/S
	elif R[0][0]>R[1][1] and R[0][0]>R[2][2]:
		S = (np.sqrt(1.0 + (R[0][0] - R[1][1] - R[2][2])))*2.0
		qw = (R[2][1] - R[1][2])/S
		qx = 0.25*S
		qy = (R[0][1] + R[1][0])/S
		qz = (R[0][2] + R[2][0])/S
	elif R[1][1] > R[2][2]:
		S = (np.sqrt(1.0 + (R[1][1] - R[0][0] - R[2][2])))*2.0
		qw = (R[0][2] - R[2][0])/S
		qy = 0.25*S
		qx = (R[0][1] + R[1][0])/S
		qz = (R[1][2] + R[2][1])/S
	else:
		S = (np.sqrt(1.0 + (R[2][2] - R[0][0] - R[1][1])))*2.0
		qw = (R[1][0] - R[0][1])/S
		qz = 0.25*S
		qy = (R[1][2] + R[2][1])/S
		qx = (R[0][2] + R[2][0])/S	
	
	q=[qw,qx,qy,qz]
	sgn = qw/abs(qw)
	q = np.multiply(q,sgn)
	
	return q

#no need to change - part of dynamics
# calculates the rotation matric given quaternion	
def Quat2Rot(q):
	ft = np.linalg.norm(q)
	#print(q)
	q = np.multiply(q,1.0/ft)
	qh = np.zeros((3,3))
	qh[0][1] = -q[3]
	qh[0][2] = q[2]
	qh[1][2] = -q[1]
	qh[1][0] = q[3]
	qh[2][0] = -q[2]
	qh[2][1] = q[1]
	R = np.eye(3) + np.multiply(np.matmul(qh,qh),2.0) + np.multiply(np.multiply(qh,q[0]),2.0)

	return R

#no need to change - part of dynamics
# calculates phi, theta, psi given rotation matrix R
def Rot2RPY_ZXY(R):
	phi = (math.asin(R[1][2]) + 2.0*np.pi)%(2.0*np.pi)
	psi = (math.atan2(-R[1][0],R[1][1])	+ 2.0*np.pi)%(2.0*np.pi)
	theta = (math.atan2(-R[0][2],R[2][2]) + 2.0*np.pi)%(2.0*np.pi)	
	return phi, theta, psi
					
#no need to change - part of dynamics	
# calculates the rotation matrix R given phi, theta, psi	
def RPY2Rot_ZXY(phi, theta, psi):
	R = np.zeros((3,3))
	R[0][0] = (np.cos(psi))*(np.cos(theta))-(np.sin(phi))*(np.sin(theta))*(np.sin(psi))
	R[0][1] = np.cos(theta)*np.sin(psi)+np.cos(psi)*np.sin(phi)*np.sin(theta)
	R[0][2] = -np.cos(phi)*np.sin(theta)
	R[1][0] = -np.cos(phi)*np.sin(psi)
	R[1][1] = np.cos(phi)*np.cos(psi)
	R[1][2] =  np.sin(phi)
	R[2][0] =   np.cos(psi)*np.sin(theta)+np.cos(theta)*np.sin(phi)*np.sin(psi)
	R[2][1] =  np.sin(psi)*np.sin(theta)-np.cos(psi)*np.cos(theta)*np.sin(phi)
	R[2][2] = np.cos(phi)*np.cos(theta)

	return R

class state_obj(object):

	def __init__(self):
		self.pos = [0.0, 0.0, 0.0]
		self.vel = [0.0, 0.0, 0.0]
		self.rot = [0.0, 0.0, 0.0]
		self.omega = [0.0, 0.0, 0.0]

#no need to change - part of dynamics 
# calculates the state variables as object
def state2Qd(x):
	qd = state_obj()
	qd.pos = x[0:3]
	qd.vel = x[3:6]
	#print(x[6:10])
	Rot = Quat2Rot(x[6:10])
	phi, theta, psi = Rot2RPY_ZXY(Rot)
	qd.rot = [phi, theta, psi]
	qd.omega = x[10:13]
	return qd

# PD controller
def controller(t, state, des_state, params):

	kp=3.0 
	kd=0.1
	ki=0.0
	Kp_phi=kp
	Kp_theta=kp
	Kp_psi=kp
	Kd_phi=kd
	Kd_theta=kd
	Kd_psi=kd
	Ki_phi=ki
	Ki_theta=ki
	Ki_psi=ki
	Kp1=10.0
	Kp2=10.0 
	Kp3=15.0 
	Kd1=20.0 
	Kd2=20.0
	Kd3=20.0
	Ki1=0.0 
	Ki2=0.0 
	Ki3=0.0
	Kpz=20.0 
	Kvz=20.0 
	Kiz=20.0
	r1_des_dotdot= des_state.acc[0] + Kd1*(des_state.vel[0] - state.vel[0]) + Kp1*(des_state.pos[0] - state.pos[0])# + Ki1*err1[0]
	r2_des_dotdot= des_state.acc[1] + Kd2*(des_state.vel[1] - state.vel[1]) + Kp2*(des_state.pos[1] - state.pos[1])# + Ki2*err1[1]
	r3_des_dotdot= des_state.acc[2] + Kd3*(des_state.vel[2] - state.vel[2]) + Kp3*(des_state.pos[2] - state.pos[2])#+ Ki3*err1[2] 
 	
	F = 0.0
	F=params.mass*(params.gravity + r3_des_dotdot)
	M = np.zeros(3)
	phi_des= ((r1_des_dotdot*np.sin(des_state.yaw) - r2_des_dotdot*np.cos(des_state.yaw))/params.gravity + 6.0*np.pi)%(2.0*np.pi)
	theta_des= ((r1_des_dotdot*np.cos(des_state.yaw) + r2_des_dotdot*np.sin(des_state.yaw))/params.gravity + 6.0*np.pi)%(2.0*np.pi)
	p_des=0.0
	q_des=0.0
	psi_des=des_state.yaw
	r_des=des_state.yawdot
	d_phi = phi_des - state.rot[0]
	d_theta = theta_des - state.rot[1]
	d_psi = psi_des - state.rot[2]
	if d_phi>np.pi:
		d_phi = d_phi - 2.0*np.pi
	if d_phi<-np.pi:
		d_phi = d_phi + 2.0*np.pi
	if d_psi>np.pi:
		d_psi = d_psi - 2.0*np.pi
	if d_psi<-np.pi:
		d_psi = d_psi + 2.0*np.pi
	if d_theta>np.pi:
		d_theta = d_theta - 2.0*np.pi
	if d_theta<-np.pi:
		d_theta = d_theta + 2.0*np.pi					
	u21= Kp_phi*(d_phi) + Kd_phi*(p_des-state.omega[0])#+ Ki_phi*err2[0]
	u22= Kp_theta*(d_theta) + Kd_theta*(q_des-state.omega[1])#+ Ki_theta*err2[1]
	u23= Kp_psi*(d_psi) + Kd_psi*(r_des-state.omega[2])#+ Ki_psi*err2[2]
	M=[u21,u22,u23]
	#print(F)
	return F, M

# power thrust equation
def Power(thrust):
	power = np.zeros(4)
	power[0]=5.668*(thrust[0]**1.4468)	
	power[1]=5.668*(thrust[1]**1.4468)	
	power[2]=5.668*(thrust[2]**1.4468)	
	power[3]=5.668*(thrust[3]**1.4468)
	return power	


# returns the dervative of the state variables
def quadEOM(s, t, T, params, alpha, alpha2, alpha3):
	# define params
	current_state = state2Qd(s)	
	st_time = TIME.time()
	des_state = trajectory_desired_state(t,T,alpha,alpha2,alpha3)
	
	FT = np.zeros(3)
	F, M = controller(t, current_state, des_state, params)
	
	A = np.zeros((4,3))
	A[0][0] = 0.25
	A[0][1] = 0.0
	A[0][2] = -0.5/params.arm_length
	A[1][0] = 0.25
	A[1][1] = 0.5/params.arm_length
	A[1][2] = 0.0
	A[2][0] = 0.25
	A[2][1] = 0.0
	A[2][2] = 0.5/params.arm_length
	A[3][0] = 0.25
	A[3][1] = -0.5/params.arm_length
	A[3][2] = 0.0
	
	FT[0] = F
	FT[1] = M[0]
	FT[2] = M[1]
	prop_thrusts = np.matmul(A,FT)
	prop_thrusts_clamped = np.zeros(4)
	prop_thrusts_clamped[0] = max(min(prop_thrusts[0], params.maxF/4.0), params.minF/4.0)
	prop_thrusts_clamped[1] = max(min(prop_thrusts[1], params.maxF/4.0), params.minF/4.0)
	prop_thrusts_clamped[2] = max(min(prop_thrusts[2], params.maxF/4.0), params.minF/4.0)
	prop_thrusts_clamped[3] = max(min(prop_thrusts[3], params.maxF/4.0), params.minF/4.0)
	power = np.zeros(4)
	power=Power(prop_thrusts_clamped) 
	B = np.zeros((3,4))
	B[0][0] = 1.0
	B[0][1] = 1.0
	B[0][2] = 1.0
	B[0][3] = 1.0
	B[1][0] = 0.0
	B[1][1] = params.arm_length
	B[1][2] = 0.0
	B[1][3] = -params.arm_length
	B[2][0] = -params.arm_length
	B[2][1] = 0.0
	B[2][2] = params.arm_length
	B[2][3] = 0.0
	F = np.matmul(B[0],prop_thrusts_clamped)
	HT = np.matmul(B[1:3],prop_thrusts_clamped)
	M[0] = HT[0]
	M[1] = HT[1]
	x = s[0]
	y = s[1]
	z = s[2]
	xdot = s[3]
	ydot = s[4]
	zdot = s[5]
	qW = s[6]
	qX = s[7]
	qY = s[8]
	qZ = s[9]
	p = s[10]
	q = s[11]
	r = s[12] 
	quat = np.array([qW,qX,qY,qZ])
	bRw = Quat2Rot(quat)
	wRb = np.transpose(bRw)
	accel = np.multiply(( np.matmul(wRb,[0.0, 0.0, F]) - [0.0,0.0,params.mass * params.gravity]),1.0 / params.mass)# fix this shit
	K_quat = 2.0
	quaterror = 1 - (qW**2 + qX**2 + qY**2 + qZ**2)
	mt = [[0.0, -p, -q, -r],[p,  0.0, -r,  q],[q,  r,  0.0, -p],[r, -q,  p,  0.0]]
	qdot =  np.multiply(np.matmul(mt,quat),-0.5) + np.multiply(quat,K_quat*quaterror)   
	omega = np.array([p,q,r])
	pqrdot   = np.matmul(params.invI,(M - np.cross(omega, np.matmul(params.I,omega)))) 
	
	sdot = np.zeros(17)
	sdot[0]  = xdot
	sdot[1]  = ydot
	sdot[2]  = zdot
	sdot[3]  = accel[0]
	sdot[4]  = accel[1]
	sdot[5]  = accel[2]
	sdot[6]  = qdot[0]
	sdot[7]  = qdot[1]
	sdot[8]  = qdot[2]
	sdot[9] =  qdot[3]
	sdot[10] = pqrdot[0]
	sdot[11] = pqrdot[1]
	sdot[12] = pqrdot[2]
	sdot[13] = power[0]
	sdot[14] = power[1]
	sdot[15] = power[2]
	sdot[16] = power[3]
	return sdot   

# shows the original straight line motion 
def original_motion(t5,Pa,Va,Pb,Vb,):
	t=0.0
	dt=0.5
	tot_step = int(t5/dt) + 1
	pos_A=np.array(Pa)
	pos_B = np.array(Pb)
	for i in range(tot_step):
			sep = np.linalg.norm(pos_A[0:2] - pos_B[0:2])
			plt.scatter(pos_A[0], pos_A[1])
			plt.pause(0.005)		
			pos_A = pos_A + np.multiply(Va,dt)
			pos_B = pos_B + np.multiply(Vb,dt)
			#print(pos_A,t)
			t = t+ dt


# flight simulation - calculates the total energy consumed for the two uavs and the total objective function cost
def simulation(t5,Pa,Va,Pb,Vb, WP_A, WP_B, T):

	cn=1
	time_cap=[]
	cons=0.0
	des_pos=[]
	des_pos2=[]
	max_time = t5
	params = system_parameters() # import from a file
	tstep = 0.01
	cstep = 0.5
	max_iter = max_time/cstep
	nstep = int(cstep/tstep)
	time = 0.0

	# for UAV 1
	des_start = Pa# enter the arguments
	des_yaw = (math.atan2(Va[1],Va[0])+ 2.0*np.pi)%(2.0*np.pi)
	des_stop = Pa + np.multiply(Va,t5)# enter the arguments
	steps = int(max_iter*nstep)
	x0 = init_state(des_start, Va, des_yaw)
	inputs = traj_inputs(Va, t5, WP_A, T)
	alpha, alpha2, alpha3 = trajectory_generation_coef(inputs)
	x=x0
	energy=[]
	energy_sum=0.0


	des_start2 = Pb# enter the arguments
	des_yaw2 = (math.atan2(Vb[1],Vb[0])+ 2.0*np.pi)%(2.0*np.pi)
	des_stop2 = Pb + np.multiply(Vb,t5)# enter the arguments
	x02 = init_state(des_start2, Vb, des_yaw2)
	inputs2 = traj_inputs(Vb, t5, WP_B, T)
	alphab, alpha2b, alpha3b = trajectory_generation_coef(inputs2)
	x2=x02
	energy_sum2=0.0


	min_sep = np.linalg.norm(np.array(Pa[0:2])-np.array(Pb[0:2]))
	
	pos_tol = 0.01
	vel_tol = 0.01

	timeint=np.zeros(nstep+1)
	for iter in range(int(max_iter)-1): # check

		if iter ==0:
			current_state = state2Qd(x)
			desired_state = trajectory_desired_state(time,T,alpha,alpha2,alpha3)
			current_state2 = state2Qd(x2)
			desired_state2 = trajectory_desired_state(time,T,alphab,alpha2b,alpha3b)

		start_time = TIME.time()
		#xsave = odeint(quadEOM, x, timeint, args=(T, time, params, alpha, alpha2, alpha3), Dfun=None)# check
		timeint = np.linspace(time, time+cstep, nstep+1)
		#print(current_state.pos,time)
		xsave = odeint(quadEOM, x, timeint, args=(T, params, alpha, alpha2, alpha3), full_output=1, rtol=0.1, atol=0.1)
		xsave2 = odeint(quadEOM, x2, timeint, args=(T, params, alphab, alpha2b, alpha3b), full_output=1, rtol=0.1, atol=0.1)		
		x = xsave[0][nstep]
		x2 = xsave2[0][nstep]		
		current_state = state2Qd(x[0:13])
		desired_state = trajectory_desired_state(time+cstep,T,alpha,alpha2,alpha3)
		current_state2 = state2Qd(x2[0:13])
		desired_state2 = trajectory_desired_state(time+cstep,T,alphab,alpha2b,alpha3b)
		# uncomment the next three lines to see a visual of the simulation
		##plt.scatter(desired_state.pos[0], desired_state.pos[1])
		##plt.scatter(desired_state2.pos[0], desired_state2.pos[1])
		##plt.pause(0.0005)
		sep = np.linalg.norm(desired_state.pos[0:2] - desired_state2.pos[0:2]) 
		if min_sep>sep:
			min_sep = sep
		time = time + cstep	
		energy_sum =  sum(x[13:17])
		energy_sum2 =  sum(x2[13:17])
	if min_sep>1.5 and min_sep<3.0:
		cons = (3.0 - min_sep)*10
	elif min_sep < 1.5:	
		cons = (1.0 - min_sep)*1000
				
	cost = energy_sum+energy_sum2 + cons
	#print(cost)
	return cost, energy_sum, energy_sum2


class samples(object):

	def __init__(self, nmax):
		self.Pa = np.zeros((nmax,3))
		self.Va = np.zeros((nmax,3))
		self.Pb = np.zeros((nmax,3))
		self.Vb = np.zeros((nmax,3))
			
# creates a  the samples for the DOE
def create_sample_points(V_max,t_det,d_thresh,n_max):
	vec = lhs(4, samples = n_max)
	d_min = d_thresh
	Va_lb=5.0
	Va_ub=V_max-5.0
	Vb_lb=1.0
	Vb_ub=V_max
	theta_a_lb=0.0
	theta_a_ub=2.0*np.pi
	delta_theta_lb=0.1
	delta_theta_ub=1.6*np.pi
	ub=np.array([theta_a_ub, Va_ub, delta_theta_ub, Vb_ub])
	lb=np.array([theta_a_lb, Va_lb, delta_theta_lb, Vb_lb])
	inp = np.zeros((n_max, 4))
	sample = samples(n_max)
	for i in range(n_max):
		vec[i]=np.multiply(vec[i],(ub-lb)) + lb
		sample.Va[i]=np.multiply([math.cos(vec[i][0]),math.sin(vec[i][0]),0.0],vec[i][1])
		theta_A=vec[i][0]
		theta_B=theta_A+vec[i][2]
		Col_Pa= sample.Pa[i] + np.multiply(sample.Va[i],t_det)
		Col_Pb= np.multiply([math.cos(theta_B), math.sin(theta_B), 0.0],d_min) + Col_Pa
		speed_B=vec[i][3]
		sample.Vb[i]=np.multiply([math.cos(theta_B), math.sin(theta_B), 0.0],speed_B)
		sample.Pb[i]=Col_Pb - np.multiply(sample.Vb[i],t_det)
		inp[i][0] = sample.Pb[i][0] - sample.Pa[i][0]
		inp[i][1] = sample.Pb[i][1] - sample.Pa[i][1]
		inp[i][2] = sample.Vb[i][0] - sample.Va[i][0]
		inp[i][3] = sample.Vb[i][1] - sample.Va[i][1] 
	return sample, inp	

##nmax=100
##samp, inp = create_sample_points(15.0,5.0,1.5,nmax)
# calculates the fitness value for NEAT
def fitness_function(genomes, config):
	for genome_id, genome in genomes:
		genome.fitness = 0.0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		for i in range(10):
			#print(i)
			Pa = samp.Pa[i]
			Va= samp.Va[i]
			Pb = samp.Pb[i]
			Vb = samp.Vb[i]
			other_params=[5.0,10.0]
			t5 = 10.0
			input_params=[Pa,Va,Pb,Vb]
			output = net.activate(inp[i])
			print(output)
			speed_A = np.linalg.norm(Va)
			speed_B = np.linalg.norm(Vb)
			speed_ub = min((15.0 - max(speed_A,speed_B)),  min(speed_A,speed_B)) 
			if output[0] < 3.0 and output[0] > 0.0 and output[1] < speed_ub and output[1] > 0.0 and output[2] < np.pi/6 and output[2] > 0.0:
				WP_A, WP_B, T = waypoints_generation(output, input_params, other_params)
				cost = simulation(t5,Pa,Va,Pb,Vb, WP_A, WP_B, T)
			else:
				cost = 10000	
			genome.fitness = genome.fitness + cost

# Running the NEAT
def run(config_file):
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)			
	p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	p.add_reporter(neat.Checkpointer(5))
	winner = p.run(fitness_function, 30)
	winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


## to create surrogate model for energy
###inp1 = np.zeros((nmax*10,5))
###inp2 = np.zeros((nmax*10,5))
###op1 = np.zeros((nmax*10,1))
###op2 = np.zeros((nmax*10,1))
###n_samp = 10
###sl = 0
###for i in range(len(samp.Pa)):
###	Pa = samp.Pa[i]
###	Va= samp.Va[i]
###	Pb = samp.Pb[i]
###	Vb = samp.Vb[i]
###	other_params=[5.0,10.0]
###	t5 = 10.0
###	input_params=[Pa,Va,Pb,Vb]
###	speed_A = np.linalg.norm(Va)
###	speed_B = np.linalg.norm(Vb)
###	speed_ub = min((15.0 - max(speed_A,speed_B)),  min(speed_A,speed_B)) 
###	theta_ub = np.pi/6.0
###	time_ub = 3.0
###	vct = lhs(3, samples = n_samp)
###	UB = np.array([time_ub, speed_ub, theta_ub])
###	LB = np.array([0.0, 0.0, 0.0])
###	for j in range(n_samp):
###		vct[j]=np.multiply(vct[j],(UB-LB)) + LB
		#print(sl)
		
###		inp1[sl][0] = Va[0]
###		inp1[sl][1] = Va[1]
###		inp1[sl][2] = vct[j][0]
###		inp1[sl][4] = vct[j][2] 
###		inp2[sl][0] = Vb[0]
###		inp2[sl][1] = Vb[1]
###		inp2[sl][2] = vct[j][0]
###		inp2[sl][4] = vct[j][2] 
###		if speed_A > speed_B:
###			inp1[sl][3] = vct[j][1]
###			inp2[sl][3] = -vct[j][1]
###		else:
###			inp1[sl][3] = -vct[j][1]
###			inp2[sl][3] = vct[j][1]	
###		WP_A, WP_B, T = waypoints_generation(vct[j], input_params, other_params)
###		cost ,energy_sum, energy_sum2 = simulation(t5,Pa,Va,Pb,Vb, WP_A, WP_B, T)
###		print('sample :', sl+1)
		#print(inp1[sl])###
###		op1[sl] = energy_sum
###		op2[sl] = energy_sum2

###		with open('input.csv','a') as fin:	
###			writer_in=csv.writer(fin)
			#writer.writerow([])
			#print(inp1[sl])
###			writer_in.writerow(inp1[sl])
###			writer_in.writerow(inp2[sl])
			#writer.writerow(np.array([inp2[sl],op2[sl]]))
###		with open('output.csv','a') as fout:		
###			writer_out=csv.writer(fout)
			#writer.writerow([])
			#print(inp1[sl])
###			writer_out.writerow(op1[sl])
###			writer_out.writerow(op2[sl])					 
###		sl = sl+1		


##if __name__ == '__main__':
##	local_dir = os.path.dirname(os.path.abspath(__file__))
##	print(local_dir)
##	config_path = os.path.join(local_dir, 'config_feedforward.py')
##	run(config_path)	



##other_params=[5.0,10.0]

##Pa = samp.Pa[1]
##Va= samp.Va[1]
##Pb = samp.Pb[1]
##Vb = samp.Vb[1]
##t5=10.0
##input_params=[Pa,Va,Pb,Vb]
##action=[2.0663,1.0,0.0888]
##WP_A, WP_B, T = waypoints_generation(action, input_params, other_params)
##print(WP_A,T)
##strt_time = TIME.time() 
##cost = simulation(t5,Pa,Va,Pb,Vb, WP_A, WP_B, T)
##print(cost)
##print(TIME.time()-strt_time)
##original_motion(t5,Pa,Va,Pb,Vb,)

