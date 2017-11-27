from Trajectory_generator import trajectory_generation_coef, trajectory_desired_state, desired_state
import numpy as np
from UAV_COL_Neat import waypoints_generation, create_sample_points, system_parameters, traj_inputs
import neat
import matplotlib.pyplot as plt
from drawnow import drawnow
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.ensemble import RandomForestRegressor 
import csv
import os
import math
plt.axis([-60, 60, -60, 60])
plt.ion()

nmax=500
samp, inp = create_sample_points(15.0,5.0,1.5,nmax)


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

sum(abs(mod.predict(X[1951:2001])- Y[1951:2001]))

def simulation(t5,Pa,Va,Pb,Vb, WP_A, WP_B, T, output):
	inp1 = np.zeros((1,5))
	inp2 = np.zeros((1,5))
	speed_A = np.linalg.norm(Va)
	speed_B = np.linalg.norm(Vb)
	inp1[0][0] = Va[0]
	inp1[0][1] = Va[1]
	inp1[0][2] = output[0]
	inp1[0][4] = output[2] 
	inp2[0][0] = Vb[0]
	inp2[0][1] = Vb[1]
	inp2[0][2] = output[0]
	inp2[0][4] = output[2] 
	if speed_A > speed_B:
		inp1[0][3] = output[1]
		inp2[0][3] = -output[1]
	else:
		inp1[0][3] = -output[1]
		inp2[0][3] = output[1]	

	energy_A = mod.predict(inp1[0:1])
	energy_B = mod.predict(inp2[0:1])
	cn=1
	time_cap=[]
	cons=0.0
	des_pos=[]
	des_pos2=[]
	max_time = t5
	params = system_parameters() # import from a file
	tstep = 0.01
	cstep = 0.25
	max_iter = max_time/cstep
	nstep = int(cstep/tstep)
	time = 0.0

	# for UAV 1
	des_start = Pa# enter the arguments
	des_yaw = (math.atan2(Va[1],Va[0])+ 2.0*np.pi)%(2.0*np.pi)
	des_stop = Pa + np.multiply(Va,t5)# enter the arguments
	steps = int(max_iter*nstep)
	#x0 = init_state(des_start, Va, des_yaw)
	inputs = traj_inputs(Va, t5, WP_A, T)
	alpha, alpha2, alpha3 = trajectory_generation_coef(inputs)
	#x=x0
	#energy=[]
	#energy_sum=0.0


	des_start2 = Pb# enter the arguments
	des_yaw2 = (math.atan2(Vb[1],Vb[0])+ 2.0*np.pi)%(2.0*np.pi)
	des_stop2 = Pb + np.multiply(Vb,t5)# enter the arguments
	#x02 = init_state(des_start2, Vb, des_yaw2)
	inputs2 = traj_inputs(Vb, t5, WP_B, T)
	alphab, alpha2b, alpha3b = trajectory_generation_coef(inputs2)
	#x2=x02
	#energy_sum2=0.0


	min_sep = np.linalg.norm(np.array(Pa[0:2])-np.array(Pb[0:2]))
	
	#pos_tol = 0.01
	#vel_tol = 0.01

	timeint=np.zeros(nstep+1)
	for iter in range(int(max_iter)-1):
		desired_state = trajectory_desired_state(time,T,alpha,alpha2,alpha3)
		desired_state2 = trajectory_desired_state(time,T,alphab,alpha2b,alpha3b)
		sep = np.linalg.norm(desired_state.pos[0:2] - desired_state2.pos[0:2]) 
		##plt.scatter(desired_state.pos[0], desired_state.pos[1])
		##plt.scatter(desired_state2.pos[0], desired_state2.pos[1])
		##plt.pause(0.0005)		
		#print(sep)

		if min_sep>sep:
			min_sep = sep
		time = time + cstep	
	if min_sep>1.5 and min_sep<3.0:
		cons = (3.0 - min_sep)*10
	elif min_sep < 1.5:	
		cons = (1.5 - min_sep)*10000000
	cost = energy_A+energy_B + cons
	##print(min_sep)
	##plt.clf()
	return cost, energy_A+energy_B, min_sep										




def fitness_function(genomes, config):
	ft=0
	for genome_id, genome in genomes:
		print(ft)
		ft=ft+1
		genome.fitness = 0.0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		for i in range(nmax):
			#print(i)
			Pa = samp.Pa[i]
			Va= samp.Va[i]
			Pb = samp.Pb[i]
			Vb = samp.Vb[i]
			other_params=[5.0,10.0]
			t5 = 10.0
			input_params=[Pa,Va,Pb,Vb]
			output = net.activate(inp[i])
			
			speed_A = np.linalg.norm(Va)
			speed_B = np.linalg.norm(Vb)
			speed_ub = min((15.0 - max(speed_A,speed_B)),  min(speed_A,speed_B)) 
			lb = np.array([0.01, 0.01, 0.01])
			ub = np.array([3.0, speed_ub, np.pi/6.0])
			output = np.multiply(output,(ub-lb)) + lb
			#print(output)
			if output[0] <= 3.0 and output[0] >= 0.001 and output[1] <= speed_ub and output[1] >= 0.001 and output[2] <= np.pi/6.0 and output[2] >= 0.001:
				WP_A, WP_B, T = waypoints_generation(output, input_params, other_params)
				cst, enrg, min_sep = simulation(t5,Pa,Va,Pb,Vb, WP_A, WP_B, T, output)
				cost = -float(cst)
				#print(0)
			else:
				print(1000000000000000000000)
				cost = -1000000000000	
			#print(genome.fitness)	
			genome.fitness = genome.fitness + cost
#writer = open('result.csv', 'a')
def run(config_file):
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)			
	p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	p.add_reporter(neat.Checkpointer(5))
	winner = p.run(fitness_function, 25)
	winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
	print(winner_net)	
	samp2, inp2 = create_sample_points(15.0,5.0,1.5,500)
	for i in range(500):
		#print(i)
		Pa = samp2.Pa[i]
		Va= samp2.Va[i]
		Pb = samp2.Pb[i]
		Vb = samp2.Vb[i]
		other_params=[5.0,10.0]
		t5 = 10.0
		input_params=[Pa,Va,Pb,Vb]
		action = winner_net.activate(inp2[i])
		
		speed_A = np.linalg.norm(Va)
		speed_B = np.linalg.norm(Vb)
		WP_A, WP_B, T = waypoints_generation(action, input_params, other_params)
		cst, enrg, min_sep = simulation(t5,Pa,Va,Pb,Vb, WP_A, WP_B, T, action)
		wrt = np.array([Pa, Va, Pb, Vb, cst, enrg, min_sep])
		#print(wrt)
		with open('result.csv','a') as fin:	
			writer_in=csv.writer(fin)
			#writer.writerow([])
			#print(inp1[sl])
			writer_in.writerow(wrt)
			#writer_in.writerow(inp2[sl])

					
 	








#if __name__ == '__main__':
local_dir = os.path.dirname(os.path.abspath(__file__))
#print(local_dir)
config_path = os.path.join(local_dir, 'config_feedforward.py')
run(config_path)
