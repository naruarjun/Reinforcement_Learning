from __future__ import division, print_function
import tensorflow as tf
import numpy as np 
import gym
import random
import sys
import time
from optparse import OptionParser
import gym_minigrid
import threading
"""
This implementation is done on multiroom environment of gym-minigrid
Repository link-> https://github.com/maximecb/gym-minigrid.git
"""
"""
This A3C implementation of proximal policy optimization(PPO) 
has 1 main actor that performs the weight updates and 4 workers 
that collect experience.The 4 workers will run on different threads at the same time
"""
rewardeach = 0
number_of_agents=4
epsi=0.1
transition = []
states = []
rewards = []
transitions = []
totalreward = 0
actions = []
returns = []
Flag = False
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

parser = OptionParser()
parser.add_option(
	"-e",
	"--env-name",
	dest="env_name",
	help="gym environment to load",
	default='MiniGrid-MultiRoom-N6-v0'
)
(options, args) = parser.parse_args()
env = gym.make('MiniGrid-Empty-6x6-v0')
env.reset()
renderer = env.render('human')
def keyDownCb(keyName):
	global Flag,states,actions,transitions
	k=0
	if keyName == 'LEFT':
		k=0
		action = env.actions.left
	elif keyName == 'RIGHT':
		k=1
		action = env.actions.right
	elif keyName == 'UP':
		k=2
		action = env.actions.forward
	elif keyName == 'SPACE':
		action = env.actions.toggle
		k=3
	else:
		print("unknown key %s" % keyName)
		return
	obs, reward, done, info = env.step(action)
	if done:
		print('done!')
		Flag=True
		env.reset()
	reward = reward-1
	states.append(obs)
	action_dummy=np.zeros(4)
	action_dummy[k]=1
	actions.append(action_dummy)
	transitions.append((obs,reward,action_dummy))
def keyDownC(keyName):
	action = 0
	k=0
	if keyName == 0:
		k=0
		action = env.actions.left
	elif keyName == 1:
		k=1
		action = env.actions.right
	elif keyName == 2:
		k=2
		action = env.actions.forward
	elif keyName == 3:
		action = env.actions.toggle
		k=3
	elif keyName == 4:
		sys.exit(0)
	else:
		print("unknown key %s" % keyName)
		return
	return action
renderer.window.setKeyDownCb(keyDownCb)
"""
The main actor in A3C framwork will be the brain
This will also provide the neccessary network for the 
workers to calculate probabilities and stochastically take actions
"""
class brain:
	def __init__(self,scope):
		with tf.variable_scope(scope):
			self.env= gym.make('MiniGrid-Empty-6x6-v0')
			self.lstm_size=128
			self.batch_size=1
			self.states=[]
			self.transitions = []
			self.actions=[]
			self.good_prob=[]
			self.good=[]
			self.returns_vector=[]
			self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
			self.state_policy = tf.placeholder(tf.float32,[None,7,7,3],name="state")
			self.actual_value = tf.placeholder("float",[None,1])
			init=tf.truncated_normal([3,3,3,16],stddev=0.1)
			self.params = tf.get_variable("policy_parameters",initializer=init)
			init=tf.truncated_normal([16],stddev=0.1)
			self.biases = tf.get_variable("biases",initializer=init)
			init=tf.truncated_normal([64,4],stddev=0.1)
			self.params2 = tf.get_variable("policy_parameters2",initializer=init)
			init=tf.truncated_normal([4],stddev=0.1)
			self.biases2 = tf.get_variable("biases2",initializer=init)
			init=tf.truncated_normal([64,1],stddev=0.1)
			self.params7 = tf.get_variable("policy_parameters7",initializer=init)
			init=tf.truncated_normal([1],stddev=0.1)
			self.biases7 = tf.get_variable("biases7",initializer=init)
			self.action = tf.placeholder(tf.float32,[None,4])
			self.how_good = tf.placeholder(tf.float32,[None,1])
			self.alpha=0.1
			self.linear1 = conv2d(self.state_policy,self.params)+self.biases
			self.linear1 = tf.maximum(self.linear1,0.1*self.linear1) 
			self.linear1 = tf.reshape(self.linear1,[-1,7*7*16])
			self.linear1 = tf.expand_dims(self.linear1,axis=0)
			self.output, self.state = tf.nn.dynamic_rnn(self.lstm,self.linear1,dtype=tf.float32)
			self.output = tf.maximum(self.output,self.alpha*self.output)
			self.loss=0
			self.output = tf.squeeze(self.output,0)
			self.outputp = self.output[:,:64]
			self.output_v = self.output[:,64:]
			self.linear_p = tf.matmul(self.outputp,self.params2)+self.biases2
			self.probabilities=tf.nn.softmax(self.linear_p)
			self.linear_v=tf.matmul(self.output_v,self.params7)+self.biases7
			self.error = self.linear_v-self.actual_value
			self.entropy = -tf.reduce_sum(self.probabilities*tf.log(self.probabilities))
			self.good_probabilities=tf.reduce_sum(tf.multiply(self.probabilities,self.action),reduction_indices=[1])
			self.good_probabilities = tf.expand_dims(self.good_probabilities,axis=1)
			self.probability_old=tf.placeholder(tf.float32,[None,1])
			self.goodness=tf.div(self.good_probabilities,self.probability_old)
			self.clip = tf.clip_by_value(self.goodness,1+0.2,1-0.2)
			self.clipped = tf.multiply(self.clip,self.how_good)
			self.goodness=tf.multiply(self.goodness,self.how_good)
			self.goodness = tf.minimum(self.goodness,self.clipped)
			self.loss = -tf.reduce_sum(self.goodness)-0.1*self.entropy
			self.loss_value = tf.nn.l2_loss(tf.reduce_sum(self.error))
			self.optimizer_value=tf.train.AdamOptimizer(1e-5).minimize(self.loss_value)
			self.optimizer = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
			self.variablevalues = tf.trainable_variables()
			self.variablenames = [v.name for v in tf.trainable_variables()]
			tf.summary.scalar("loss",self.loss)
			self.merged = tf.summary.merge_all()
			self.init=tf.global_variables_initializer()
			self.graph = tf.get_default_graph()
"""
	def train(self,sess):
		_,loss_value=sess.run([self.optimizer_value,self.loss_value],feed_dict={self.state_policy:self.states,self.actual_value:self.returns_vector})
		_,loss_result,summ=sess.run([self.optimizer,self.loss,self.merged],feed_dict={self.state_policy:self.states,self.action:self.actions,self.probability_old:self.good_prob,self.how_good:self.good})		
"""
"""
Used to create Workers
Each instance of this object creates a new thread and the run method
runs through one episode
"""
class Worker(threading.Thread):
	def __init__(self,name):
		threading.Thread.__init__(self)
		self.name=name
		self.transitions = []
		self.states = []
		self.good=[]
		self.returns=[]
		self.returns_vector=[]
		self.network_variables = []
		self.actions=[]
	def run(self):
		local_ppo= brain(self.name)
		with tf.Session(graph=local_ppo.graph) as sess:
			count=0
			for k in local_ppo.variablenames:
				t_name = local_ppo.graph.get_tensor_by_name(k)
				op=tf.assign(t_name,self.network_variables[count])
				sess.run(op)
				count=count+1
			local_ppo.env.seed(2)
			observation = local_ppo.env.reset()
			observation_copy = observation
			local_ppo.env.seed(2)
			old_observation=local_ppo.env.reset()
			old_observation_copy = old_observation
			totalreward = 0
			flag =False
			distance_to_gates_new=0
			distance_to_gates_old=0
			rewardsqwerty=0
			for n in range(120):
				obs_vector = np.zeros([1,7,7,3])
				obs_vector[0]=observation
				#Calculating probabiities
				probs = sess.run(local_ppo.probabilities,feed_dict={local_ppo.state_policy:obs_vector})
				p=np.array(probs)
				#print(probs)
				arr=[0,1,2,3]
				h=random.uniform(0,1)
				action_take = np.random.choice(arr,1,p=np.reshape(probs,[4,]))
				for q in range(7):
					for g in range(7):
						if observation[q,g,0]==2:
							observation[q,g,0]=1000
						if observation[q,g,0]==7:
							observation[q,g,0]=4000
				self.states.append(observation)
				action_taken = np.zeros(4)
				action_taken[action_take]=1
				self.actions.append(action_taken)
				action_dummy = action_take
				action_take=keyDownC(action_take)
				old_observation=observation
				old_observation_copy=observation_copy
				observation,reward,done,info=local_ppo.env.step(action_take)
				#local_ppo.env.render('human')
				observation_copy = observation
				number=0
				for z in range(len(observation_copy)):
					for x in range(len(observation_copy[0])):
						if observation_copy[z,x,0]==2 and observation_copy[z,x,2]==0:
							number=1
							distance_to_gates_new = (3-z)**2+(6-x)**2
				for q in range(7):
					for g in range(7):
						if observation[q,g,0]==2:
							observation[q,g,0]=1000
						if observation[q,g,0]==7:
							observation[q,g,0]=4000
				for q in range(7):
					for g in range(7):
						if observation[q,g,0]==2:
							observation[q,g,0]=1000
						if observation[q,g,0]==7:
							observation[q,g,0]=4000
				rewardsqwerty=reward
				#env.render('human')
				observation_copy = observation
				if old_observation_copy[3,5,2]==0 and observation_copy[3,5,2]==1 and old_observation_copy[3,5,0]==232 and observation_copy[3,5,0]==232:
					reward=reward+40
				if number>0:
					if (distance_to_gates_new<distance_to_gates_old):
						reward=reward+3
					if (distance_to_gates_new>distance_to_gates_old):
						reward=reward-3	
				distance_to_gates_old=distance_to_gates_new
				reward = reward-1
				self.transitions.append((old_observation,reward,action_take))
				totalreward+=reward
				if done:
					if rewardsqwerty>900:
						print(reward)
					break	
			for index,trans in enumerate(self.transitions):
				obs,reward,action_take=trans
				future_trans = len(transitions)-index
				gamma=1
				future_return=0
				for index2 in range(future_trans):
					future_return += transitions[index2+index][1]*gamma
					gamma=gamma*0.999
				obs_vector = np.expand_dims(obs,axis=0)
				predicted_value = sess.run([local_ppo.linear_v],feed_dict={local_ppo.state_policy:obs_vector})[0][0]
				goods = future_return-predicted_value
				self.returns.append(future_return)
				self.good.append(goods)
			self.returns_vector = np.expand_dims(self.returns, axis=1)
#FZHMDPWV
"""
Each worker stores states actions and rewards as instance variables 
and in the end of one episode for each worker all experience is combined(4 episodes in total)
and update is performed on the main_actor
"""

workers=[]
main_actor = brain("The_dude")
with tf.Session(graph=main_actor.graph) as sess:
	#train_writer = tf.summary.FileWriter('/home/naruarjun/taskphaseproj_naru/gym-minigrid/tensormodel4',tf.get_default_graph())
	sess.run(main_actor.init)
	for l in range(2000):
		#for i in range(number_of_agents):
		#	workers.append(Worker("AGENT%d"%(i)))
		worker1 = Worker("AGENT%d"%(l+10))
		worker2 = Worker("AGENT%d"%(l+20))
		worker3 = Worker("AGENT%d"%(l+30))
		worker4 = Worker("AGENT%d"%(l+40))
		workers= [worker1,worker2,worker3,worker4]
		if l==0:
			old_policy=sess.run(main_actor.variablenames)
			new_policy=sess.run(main_actor.variablenames)
			#print(old_policy)
		worker_variable = []
		for worker in workers:
			worker_variable=[]
			for k in old_policy:
				worker_variable.append(k)
			worker.network_variables=worker_variable
		for worker in workers: 
			worker.start()
			worker.join()
		episodes=[]
		actions=[]
		transitions=[]
		goodp=[]
		returns_vectorp=[]
		for j in range(len(workers)):
			if j==0:
				#print(np.array(workers[j].states).shape)
				episodes = np.array(workers[j].states)
				actionsp = np.array(workers[j].actions)
				transitionsp = np.array(workers[j].transitions)
				goodsp = np.array(workers[j].good)
				returns_vectorp = np.array(workers[j].returns_vector)
			else:
				episodes = np.vstack((episodes,workers[j].states))
				actionsp = np.vstack((actionsp,workers[j].actions))
				transitions = np.vstack((transitionsp,workers[j].transitions))
				goodsp = np.vstack((goodsp,workers[j].good))
				returns_vectorp = np.vstack((returns_vectorp,workers[j].returns_vector))
		#print(episodes)
		prob_old = sess.run(main_actor.good_probabilities,feed_dict={main_actor.state_policy:episodes,main_actor.action:actionsp})
		for x,c in zip(main_actor.variablenames, new_policy):
			t_name = main_actor.graph.get_tensor_by_name(x)
			op=tf.assign(t_name,c)
			sess.run(op)
		main_actor.states=episodes
		main_actor.actions = actionsp
		main_actor.transitions = transitionsp
		main_actor.good_prob = prob_old
		main_actor.good = goodsp
		main_actor.returns_vector = returns_vectorp
		old_policy=new_policy
		_,loss_value=sess.run([main_actor.optimizer_value,main_actor.loss_value],feed_dict={main_actor.state_policy:episodes,main_actor.actual_value:returns_vectorp})
		_,loss_result,summ=sess.run([main_actor.optimizer,main_actor.loss,main_actor.merged],feed_dict={main_actor.state_policy:episodes,main_actor.action:actionsp,main_actor.probability_old:prob_old,main_actor.how_good:goodsp})
		print(loss_result)
		new_policy=sess.run(main_actor.variablenames)
		for x,c in zip(main_actor.variablenames, old_policy):
			t_name = main_actor.graph.get_tensor_by_name(x)
			op=tf.assign(t_name,c)
			sess.run(op)
		print("i:%d       reward:%d"%(l,rewardeach))
		print("yay")