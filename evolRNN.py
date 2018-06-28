import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import gym

#if game changes, change input size; may have to reshape input

num_inp = 4
batch_size = 1
num_nodes = 3
num_time_steps = 4
num_out = 2

num_save = 10
evol_per_graph = 10
num_graphs = num_save*evol_per_graph

epochs = 400
num_games = 4

steps = 500
env_name = 'CartPole-v0'
env_name1 = 'Assault-v0'

def weight_list(num_inp,num_nodes,num_time_steps):
	weights = []
	for i in range(num_time_steps):
		if(i==0):
			weights.append(tf.Variable(tf.truncated_normal([num_inp,num_nodes],stddev=.1)))
		else:
			weights.append(tf.Variable(tf.truncated_normal([num_nodes,num_nodes],stddev=.1)))
	return weights

def bias_list(num_inp,num_nodes,num_time_steps):
	bias = []
	for i in range(num_time_steps):
		bias.append(tf.Variable(tf.truncated_normal([num_nodes],stddev=.1)))
	return bias

def rnn_graph(w,b,x):
	y_comp = 0
	y_layer = []
	for i in range(len(w)):
		if(i==0):
			y_comp = tf.tanh(tf.matmul(x,w[i]) + b[i])
		else:
			y_comp = tf.tanh(tf.matmul(y_layer[i-1],w[i]) + b[i] + y_comp)
		y_layer.append(y_comp)		
	return y_comp

def add_layers(nodes_per_lay,num_lay,lay_1):
	w = tf.Variable(tf.random_uniform([nodes_per_lay,nodes_per_lay]))
	b = tf.Variable(tf.random_uniform([nodes_per_lay]))
	y = tf.nn.relu(tf.matmul(lay_1,w)+b)
	if num_lay == 0:
		return y
	else:
		return add_layers(nodes_per_lay,num_lay-1,y)

def find_best_graphs(r_tot_list,graph_objects,graph_outputs,num_save):
	reward_arr = np.array(r_tot_list)
	graph_obj_arr = np.array(graph_objects)
	graph_outputs_arr = np.array(graph_outputs)
	
	indices = reward_arr.argsort()
	indices = np.flip(indices,0)

	sorted_reward = reward_arr[indices]
	sorted_graph_obj = graph_obj_arr[indices]
	sorted_graph_out = graph_outputs_arr[indices]

	return zip(sorted_reward[0:num_save],sorted_graph_obj[0:num_save],sorted_graph_out[0:num_save])

def Create_graph_list(batch_size,num_inp,num_nodes,num_out,num_time_steps,num_graphs):
	x = tf.placeholder(tf.float32,shape=[batch_size,num_inp],name="X")

	W_out = tf.Variable(tf.truncated_normal([num_nodes,num_out],stddev=.1))
	b_out = tf.Variable(tf.truncated_normal([1,num_out],stddev=.1))

	weights = weight_list(num_inp,num_nodes,num_time_steps)
	biases = bias_list(num_inp,num_nodes,num_time_steps)

	y1 = rnn_graph(weights,biases,x)
	y4 = tf.nn.softmax(tf.matmul(y1,W_out)+b_out)
	return y4



graph_objects = [tf.Graph() for i in range(num_graphs)]
graph_outputs = []
for i in range(num_graphs):
	with graph_objects[i].as_default():
		hold_graph_output = Create_graph_list(batch_size,num_inp,num_nodes,num_out,num_time_steps,num_graphs)
	graph_outputs.append(hold_graph_output)


best_reward = []
best_graph_objects = []
best_graph_outputs = []

#repeats evolutionary process
for ep in range(epochs):
	r_tot_list = []
	#run through each graph
	for i in range(num_graphs):
		with tf.Session(graph = graph_objects[i]) as sess:
			sess.run(tf.global_variables_initializer())
			
			#make each graph play through number of games
			for game in range(num_games):
				env = gym.make(env_name)

				j = 0
				done = False
				r_tot = 0

				observation = env.reset()

				#finish single playthrough of single graph
				while(not done and j<steps):
					j=j+1
					x = tf.get_default_graph().get_tensor_by_name('X:0')
					z = sess.run(tf.argmax(graph_outputs[i][0]),feed_dict={x:observation.reshape(1,num_inp)})

					observation, reward, done, info = env.step(z)
					r_tot = r_tot + reward
					#time.sleep(.05)
					#env.render()

			r_tot_list.append(r_tot/num_games)

	#get best rewards and graphs
	best_reward.extend(r_tot_list)
	best_graph_objects.extend(graph_objects)
	best_graph_outputs.extend(graph_outputs)

	zipped_best_graphs = find_best_graphs(best_reward,best_graph_objects,best_graph_outputs,num_save)
	best_reward,best_graph_objects,best_graph_outputs = zip(*zipped_best_graphs)

	best_reward = list(best_reward)
	best_graph_objects = list(best_graph_objects)
	best_graph_outputs = list(best_graph_outputs)

	print("epoch: "+str(ep))
	print(best_reward)


	#evolvs NN
	for i in range(num_graphs):
		with tf.Session(graph = graph_objects[i]) as sess:
			sess.run(tf.global_variables_initializer())
			if(ep%10== 0):
				saver = tf.train.Saver()
				saver.save(sess,"tmp/rnn_model.ckpt")
			for j in tf.trainable_variables():
				rand_arr = np.random.uniform(low=-0.01,high=.01,size=j.shape)
				weights_new = rand_arr+j.eval()
				assign_op = j.assign(weights_new)
				sess.run(assign_op)





	




