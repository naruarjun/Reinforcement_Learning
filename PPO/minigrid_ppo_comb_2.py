from __future__ import division, print_function
import tensorflow as tf
import numpy as np 
import gym
import random
import sys
import time
from optparse import OptionParser
import gym_minigrid
"""
This implementation is done on multiroom environment of gym-minigrid
Repository link-> https://github.com/maximecb/gym-minigrid.git
"""
"""
PPO implementation on gym -minigrid
"""
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
lstm_size=128
batch_size=1
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
lstm2 = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
lstm3 = tf.nn.rnn_cell.BasicLSTMCell(64)
"""
The network for estimation of the policy and the value function
"""
def policy_gradient():
	global lstm,hidden_state,current_state,state,lstm_size
	with tf.variable_scope("policy"):
		state_policy = tf.placeholder(tf.float32,[None,7,7,3])
		actual_value = tf.placeholder("float",[None,1])
		inde = tf.placeholder(tf.int32,[1])
		an=inde[0]
		init=tf.truncated_normal([3,3,3,16],stddev=0.1)
		params = tf.get_variable("policy_parameters",initializer=init)
		init=tf.truncated_normal([16],stddev=0.1)
		biases = tf.get_variable("biases",initializer=init)
		init=tf.truncated_normal([64,4],stddev=0.1)
		params2 = tf.get_variable("policy_parameters2",initializer=init)
		init=tf.truncated_normal([4],stddev=0.1)
		biases2 = tf.get_variable("biases2",initializer=init)
		init=tf.truncated_normal([64,1],stddev=0.1)
		params7 = tf.get_variable("policy_parameters7",initializer=init)
		init=tf.truncated_normal([1],stddev=0.1)
		biases7 = tf.get_variable("biases7",initializer=init)
		action = tf.placeholder(tf.float32,[None,4])
		how_good = tf.placeholder(tf.float32,[None,1])
		alpha=0.1
		linear1 = conv2d(state_policy,params)+biases
		linear1 = tf.maximum(linear1,0.1*linear1) 
		linear1 = tf.reshape(linear1,[-1,7*7*16])
		linear1 = tf.expand_dims(linear1,axis=0)
		output, state = tf.nn.dynamic_rnn(lstm,linear1,dtype=tf.float32)
		output = tf.maximum(output,alpha*output)
		loss=0
		output = tf.squeeze(output,0)
		outputp = output[:,:64]
		output_v = output[:,64:]
		linear_p = tf.matmul(outputp,params2)+biases2
		probabilities=tf.nn.softmax(linear_p)
		linear_v=tf.matmul(output_v,params7)+biases7
		error = linear_v-actual_value
		entropy = -tf.reduce_sum(probabilities*tf.log(probabilities))
		good_probabilities=tf.reduce_sum(tf.multiply(probabilities,action),reduction_indices=[1])
		good_probabilities = tf.expand_dims(good_probabilities,axis=1)
		print(good_probabilities)
		probability_old=tf.placeholder(tf.float32,[None,1])
		goodness=tf.div(good_probabilities,probability_old)
		clip = tf.clip_by_value(goodness,1+0.2,1-0.2)
		goodness=tf.multiply(goodness,how_good)
		clipped = tf.multiply(clip,how_good)
		goodness = tf.minimum(clipped,goodness)
		loss = -tf.reduce_sum(goodness)-0.1*entropy
		loss_value = tf.nn.l2_loss(tf.reduce_sum(error))
		optimizer_value=tf.train.AdamOptimizer(1e-5).minimize(loss_value)
		optimizer = tf.train.AdamOptimizer(1e-5).minimize(loss)
		tf.summary.scalar("loss",loss)
		merged = tf.summary.merge_all()
		return probabilities,state_policy,action,how_good,optimizer,loss,params,output,probability_old,good_probabilities,inde,linear_v,actual_value,optimizer_value,loss_value,merged
parser = OptionParser()
parser.add_option(
	"-e",
	"--env-name",
	dest="env_name",
	help="gym environment to load",
	default='MiniGrid-MultiRoom-N6-v0'
)
(options, args) = parser.parse_args()
env = gym.make(options.env_name)
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
policy_grad = policy_gradient()
saver = tf.train.Saver()
with tf.Session() as sess:
	train_writer = tf.summary.FileWriter('/home/naruarjun/taskphaseproj_naru/gym-minigrid/tensormodel4',tf.get_default_graph())
	sol=0
	sess.run(tf.global_variables_initializer())
	sol=0
	for m in range(1):
		for i in range(10000):
			"""
			Running 10000 episodes
			"""
			if m!=0:
				saver.restore(sess, "/home/naruarjun/taskphaseproj_naru/gym-minigrid/model15.ckpt")
			elif m==0 and i!=0:
				saver.restore(sess, "/home/naruarjun/taskphaseproj_naru/gym-minigrid/model15.ckpt")
			p1_prob,p1_state,p1_action,p1_how_good,p1_optimizer,p1_loss,p1_params,p1_stata,p1_prob_ratio,p1_good_probabilities,p1_inde,v1_pred_value,v1_value,v1_optimizer,v1_loss,p1_summary_op = policy_grad
			env.seed(2)
			observation = env.reset()
			observation_copy = observation
			env.seed(2)
			old_observation=env.reset()
			old_observation_copy = old_observation
			transition = []
			states = []
			rewards = []
			transitions = []
			totalreward = 0
			actions = []
			returns = []
			good = []
			flag=False
			distance_to_gates_new=0
			distance_to_gates_old=0
			rewardsqwerty=0
			if epsi<0.95:
				epsi = epsi +1./((i/50)+30000)
			else:
				epsi = 0.95
			for n in range(120):
				"""
				Code to play the game and collect data is commented out
				"""
				"""
				if i<10:
					while True:
						env.render('human')
						time.sleep(0.01)
						if Flag==True:
							print('im her')
							break
						# If the window was closed
						if renderer.window == None:
							break
					break
				"""
				obs_vector = np.zeros([1,7,7,3])
				obs_vector[0]=observation
				probs = sess.run(p1_prob,feed_dict={p1_state:obs_vector})
				p=np.array(probs)
				arr=[0,1,2,3]
				h=random.uniform(0,1)
				action_take = np.random.choice(arr,1,p=np.reshape(probs,[4,]))
				for q in range(7):
					for g in range(7):
						if observation[q,g,0]==2:
							observation[q,g,0]=1000
						if observation[q,g,0]==7:
							observation[q,g,0]=4000
				states.append(observation)
				action_taken = np.zeros(4)
				action_taken[action_take]=1
				actions.append(action_taken)
				action_dummy = action_take
				action_take=keyDownC(action_take)
				old_observation=observation
				old_observation_copy=observation_copy
				observation,reward,done,info=env.step(action_take)
				env.render('human')
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
				transitions.append((old_observation,reward,action_take))
				totalreward+=reward
				if done:
					if rewardsqwerty>900:
						sol=sol+1
					break
			Flag=False
			good_prob=[]
			good_prob=sess.run(p1_good_probabilities,feed_dict={p1_state:states,p1_action:actions})
			print(np.array(good_prob).shape)
			if m!=0:
				saver.restore(sess, "/home/naruarjun/taskphaseproj_naru/gym-minigrid/model16.ckpt")
			elif m==0 and i!=0:
				saver.restore(sess, "/home/naruarjun/taskphaseproj_naru/gym-minigrid/model16.ckpt")
			state_sent=[]
			action_sent=[]		
			for _ in range(1):
				good = []
				returns = []
				retur =0
				goods=0
				for index,trans in enumerate(transitions):
					obs,reward,action_take=trans
					future_trans = len(transitions)-index
					gamma=1
					future_return=0
					for index2 in range(future_trans):
						future_return += transitions[index2+index][1]*gamma
						gamma=gamma*0.999
					obs_vector = np.expand_dims(obs,axis=0)
					predicted_value = sess.run([v1_pred_value],feed_dict={p1_state:obs_vector})[0][0]
					goods = future_return-predicted_value
					returns.append(future_return)
					good.append(goods)
					state_sent.append(states[index])
					action_sent.append(actions[index])
				returns_vector = np.expand_dims(returns, axis=1)
				save_path = saver.save(sess, "/home/naruarjun/taskphaseproj_naru/gym-minigrid/model15.ckpt")
				_,loss_value=sess.run([v1_optimizer,v1_loss],feed_dict={p1_state:states,v1_value:returns_vector})
				_,loss_result,summ=sess.run([p1_optimizer,p1_loss,p1_summary_op],feed_dict={p1_state:states,p1_action:actions,p1_prob_ratio:good_prob,p1_how_good:good})
				save_path = saver.save(sess, "/home/naruarjun/taskphaseproj_naru/gym-minigrid/model16.ckpt")
			print('maze:%d   i:%d   reward:%d   loss:%f    actual_reward:%f    sol:%d'%(m,i,totalreward,loss_result,rewardsqwerty,sol))
			train_writer.add_summary(summ, i)
"""
I am saving and restoring the models to get the models,
This is inefficient and rectified in A3C_PPO
"""