import tensorflow as tf
import numpy as np 
import gym
import random
"""
This just requires openai gym
"""
"""
I have used Monte Carlo Returns but TD or TD(lambda) can also be used
"""
"""
Network to evaluate the policy
"""
def policy_gradient():
	with tf.variable_scope("policy"):
		state_policy = tf.placeholder("float",[None,4])
		init=tf.truncated_normal([4,2],stddev=0.1)
		params = tf.get_variable("policy_parameters",initializer=init)
		action = tf.placeholder("float",[None,2])
		how_good = tf.placeholder("float",[None,1])
		linear = tf.matmul(state_policy,params)
		probabilities=tf.nn.softmax(linear)
		good_probabilities=tf.reduce_sum(tf.multiply(probabilities,action),reduction_indices=[1])
		goodness=tf.log(good_probabilities)*how_good
		loss = -tf.reduce_sum(goodness)
		tf.summary.scalar("loss",loss)
		merged = tf.summary.merge_all()
		optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
		return probabilities,state_policy,action,how_good,optimizer,loss,params,merged
"""
Network to estimate value function
"""
def value_function():
	with tf.variable_scope("value"):
		state_value = tf.placeholder("float",[None,4])
		actual_value = tf.placeholder("float",[None,1])
		init=tf.truncated_normal([4,10],stddev=0.1)
		w1 = tf.get_variable("w1",initializer=init)
		init=tf.truncated_normal([10],stddev=0.1)
		b1 = tf.get_variable("b1",initializer=init)
		init=tf.truncated_normal([10,1],stddev=0.1)
		w2 = tf.get_variable("w2",initializer=init)
		init=tf.truncated_normal([1],stddev=0.1)
		b2 = tf.get_variable("b2",initializer=init)
		l1 = tf.nn.relu(tf.matmul(state_value,w1)+b1)
		pred_value = tf.matmul(l1,w2)+b2
		error = pred_value-actual_value
		loss = tf.nn.l2_loss(error)
		optimizer_value=tf.train.AdamOptimizer(0.1).minimize(loss)
		return state_value,pred_value,actual_value,optimizer_value
def run_episode(env,policy_grad,value_grad,sess,train_writer):
	p1_prob,p1_state,p1_action,p1_how_good,p1_optimizer,p1_loss,p1_params,p1_merged = policy_grad
	v1_state,v1_pred_value,v1_value,v1_optimizer = value_grad
	observation = env.reset()
	observation = observation["image"]
	old_observation=env.reset()
	old_observation = old_observation["image"]
	transition = []
	states = []
	rewards = []
	transitions = []
	totalreward = 0
	actions = []
	returns = []
	good = []
	for n in range(1000):
		obs_vector = np.expand_dims(observation, axis=0)
		probs = sess.run(p1_prob,feed_dict={p1_state:obs_vector})
		if n%30==0:
			if probs[0][0]>probs[0][1]:
				action_take=0
			else:
				action_take=1
		else:
			action_take=0 if random.uniform(0,1)<probs[0][0] else 1
		states.append(observation)
		action_taken = np.zeros(2)
		action_taken[action_take]=1
		actions.append(action_taken)
		old_observation=observation
		observation,reward,done,info=env.step(action_take)
		observation = observation["image"]
		env.render()
		transitions.append((old_observation,reward,action_take))
		totalreward+=reward
		if done:
			break
	#This is where training happens
	for index,trans in enumerate(transitions):
		obs,reward,action_take=trans
		future_trans = len(transitions)-index
		gamma=1
		future_return=0
		for index2 in range(future_trans):
			future_return += transitions[index2+index][1]*gamma
			gamma=gamma*0.97
		obs_vector = np.expand_dims(obs, axis=0)
		predicted_value = sess.run(v1_pred_value,feed_dict={v1_state:obs_vector})[0][0]
		good.append(future_return-predicted_value)
		returns.append(future_return)
	returns_vector = np.expand_dims(returns, axis=1)
	sess.run(v1_optimizer,feed_dict={v1_state:states,v1_value:returns_vector})
	good_vector = np.expand_dims(good, axis=1)
	loss_result,_,para,summ=sess.run([p1_loss,p1_optimizer,p1_params,p1_merged],feed_dict={p1_state:states,p1_action:actions,p1_how_good:good_vector})
	train_writer.add_summary(summ, i)
	return totalreward,loss_result,para
env = gym.make('CartPole-v0')
policy_grad = policy_gradient()
value_grad = value_function()
with tf.Session() as sess:
	train_writer = tf.summary.FileWriter('/home/naruarjun/taskphaseproj_naru/gym-minigrid/tensormodel2',tf.get_default_graph())
	sess.run(tf.global_variables_initializer())
	for i in range(2000):
		rewards,loss_observed,param = run_episode(env, policy_grad, value_grad, sess,train_writer)
		if rewards>=200:
			print('%d:%f loss:%f'%(i,rewards,loss_observed))
			break
		print('%d:%f loss:%f'%(i,rewards,loss_observed))






