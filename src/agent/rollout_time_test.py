import argparse
import copy
import logging
import os
import math
import random
import time
from collections import deque

import ray
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.env.utils import get_env
from src.nn import nn_utils
from src.plt_utils import *


@ray.remote
def evaluate():
	pass

def load_model_params(sample, buffer, model_class, model_param, env_name, frame_limit, buffer_prob, env_seed=None):
	with torch.no_grad():
		device = nn_utils.get_device()
		nn_model = model_class(**model_param).to(device)
		param_dict = {name[0]: param for name, param in zip(nn_model.state_dict().items(), sample)}
		nn_model.load_state_dict(param_dict)
		print("debug tag")
		g_reward, frames = rollout(nn_model, env_name, env_seed, frame_limit, buffer_prob)
		print("debug tag")
		sum_action_prob = diversity_sampling(nn_model, buffer)
		return g_reward, frames, sum_action_prob


def rollout(nn_model, env_name, env_seed, frame_limit, buffer_prob):
	if env_seed is None:
		env_seed = 0
	device = nn_utils.get_device()
	env = get_env(env_name)
	g_reward = 0
	done = True
	frames = []
	s = time.time()
	for _ in range(frame_limit):
		if done:
			state, info = env.reset(seed=env_seed)
			env_seed += 1
			done = False
		else:
			state, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			g_reward += reward
		obs = torch.tensor(state).to(device).float() / 255
		if random.random() < buffer_prob:
			frames.append(obs)
		action_prob = nn_model(obs.unsqueeze(0))
		action = torch.argmax(action_prob)
	print(f"rollout time: {time.time() - s}")
	return g_reward, frames


def diversity_sampling(nn_model, buffer):
	loader = DataLoader(buffer, 500, False)
	sum_action_prob = None
	for batch in loader:
		print(batch.shape)
		action_prob = nn_model(batch)
		action_prob = torch.sum(action_prob, dim=[0, ])
		if sum_action_prob is None:
			sum_action_prob = action_prob
		else:
			sum_action_prob = sum_action_prob + action_prob
	return sum_action_prob / len(buffer)


class DiversityWorker:
	def __init__(self, env_name, buffer_size, env_param=None):
		# buffer
		self.buffer_size = buffer_size
		self.env = get_env(env_name, env_param)
		self.env_seed = 0
		self.buffer = deque(maxlen=self.buffer_size)
		# self.env.seed(self.env_seed)
		self.env.action_space.seed(self.env_seed)
		self.env_seed += 1
		self.env.reset(seed=self.env_seed)
		self.buffer_prob = 0.2
		self.update_rate = 0.5
		
		s = time.time()
		step = 0
		while len(self.buffer) < self.buffer_size:
			step+=1
			action = self.env.action_space.sample()
			state, reward, terminated, truncated, info = self.env.step(action)
			if self.buffer_prob > random.random():
				self.buffer.append(torch.tensor(state).to(nn_utils.get_device()).float() / 255)
			done = terminated or truncated
			if done:
				# self.env.seed(self.env_seed)
				self.env.action_space.seed(self.env_seed)
				self.env_seed += 1
				self.env.reset(seed=self.env_seed)
		print(f'init buffer: {time.time()-s} seconds, {step} steps.')
		
	
	def update_buffer(self, frame_futs):
		for fut in frame_futs:
			self.buffer.extend(ray.get(fut))
		random.shuffle(self.buffer)
	
	def get_buffer_fut(self):
		return self.buffer


class NESWorker:
	def __init__(self, worker_id, random_seed, parameter_shapes, diversity_worker, rollout_params,
	             sampling_size=15, phi=0.0001, init_eta=(0.5, 0.1)):
		
		self.avgfits = []
		self.bestfits = []
		self.sampling_size = sampling_size
		self.init_eta = init_eta
		self.phi = phi
		
		self.worker_id = worker_id
		self.random_seed = random_seed
		self.parameter_shapes = parameter_shapes
		self.diversity_worker = diversity_worker
		self.rollout_params = rollout_params
		
		torch.manual_seed(self.random_seed)
		self.individual = self.init_individual()
		self.best_solution = None
		self.res_f = None
		self.res_self_d = None
		self.sum_action_prob = None
		self.samples = None
		self.step = 0
		
		# init normalized fitness by ranking
		temp = [max(math.log(self.sampling_size / 2 + 1) - math.log(i + 1), 0) for i in range(self.sampling_size)]
		temp_sum = sum(temp)
		self.normalized_fitness = [i / temp_sum - 1 / self.sampling_size for i in temp]
		self.cal_etas = lambda progress: tuple(
			eta * (math.e - math.e ** progress) / (math.e - 1) for eta in self.init_eta)
		
		self.logger = self.init_logger()
		mean_value_m = torch.mean(self.individual[0][-3]).item()
		min_value_m = torch.min(self.individual[0][-3]).item()
		max_value_m = torch.max(self.individual[0][-3]).item()
		mean_value_s = torch.mean(self.individual[1][-3]).item()
		min_value_s = torch.min(self.individual[1][-3]).item()
		max_value_s = torch.max(self.individual[1][-3]).item()
		self.log_infos = {
			"means": [(min_value_m, max_value_m, mean_value_m), ],
			"sigmas": [(min_value_s, max_value_s, mean_value_s), ]
		}
	
	
	
	@staticmethod
	def normal_sample_using_shapes(shapes, mu: float = 0, std: float = 1):
		return tuple(torch.randn(shape) * std + mu for _, shape in shapes.items())
	
	def init_individual(self):
		means = self.normal_sample_using_shapes(self.parameter_shapes)
		sigmas = self.normal_sample_using_shapes(self.parameter_shapes, 0.08, 0)
		# sigmas = [i ** 2 for i in stds]
		return means, sigmas
	
	def run_sampling(self):
		# sample parameters
		means, sigmas = self.individual
		samples = []
		for mean, sigma in zip(means, sigmas):
			std = sigma ** 0.5
			sample = [torch.randn_like(mean) * std + mean for _ in range(self.sampling_size)]
			samples.append(sample)
		
		ret = list(zip(*samples))
		return ret
	
	
	def search1(self, progress):
		samples = self.run_sampling()
		self.samples = samples
		
		buffer = self.diversity_worker()
		futs = []
		for sample in self.samples:
			futs.append(load_model_params(sample=sample, buffer_fut=buffer, **self.rollout_params))
			break
		
		
	

class ANESTrainer:
	def __init__(self, params):
		"""
		"""
		self.params = params
		# Modules
		self.nn_class = nn_utils.get_network_class(params.nn_class)
		# Training
		self.iteration = params.iterations
		self.population_size = params.population_size  # population_size
		self.sampling_size = params.sampling_size
		self.best_size = 5
		self.time_budget = params.time_budget * 60
		self.forward_worker_num = self.sampling_size * self.population_size
		self.enable_time_limit = params.enable_time_limit
		self.buffer_size = params.buffer_size
		self.buffer_updating_rate = params.buffer_updating_rate
		self.frame_limit = params.frame_limit
		self.buffer_prob = self.buffer_size * self.buffer_updating_rate / (
				self.sampling_size * self.population_size * self.frame_limit)
		
		# check env
		self.env_name = params.env_name
		self.env = get_env(self.env_name)
		
		self.model_hyper_param = {"in_channels": params.input_channels,
		                          "num_actions": int(self.env.action_space.n)}
		self.rollout_params = {'model_class': self.nn_class,
		                       'model_param': self.model_hyper_param,
		                       'env_name': self.env_name,
		                       'frame_limit': self.frame_limit,
		                       'buffer_prob': self.buffer_prob}
		
		# init normalized fitness by ranking
		self.normalized_fitnesses = [max(math.log(self.sampling_size / 2 + 1) - math.log(i + 1), 0)
		                             for i in range(self.sampling_size)]
		temp_sum = sum(self.normalized_fitnesses)
		self.normalized_fitnesses = [i / temp_sum - 1 / self.sampling_size for i in self.normalized_fitnesses]
		
		# init distribution population
		self.parameter_shapes = self.get_model_parameter_shapes()
		self.population = tuple(self.init_individual() for _ in range(self.population_size))
		
		self.search_workers = []
		self.best_solutions = []
		print(f"Iteration: {self.iteration}")
		print(f"population_size: {self.population_size}")
		print(f"sampling_size: {self.sampling_size}")
		print(f"time_budget: {self.time_budget}s")
	
	def save_models(self, model, name):
		"""保存模型的所有参数

		Args:
			model (_type_): 神经网络
			name (_type_): 文件名
		"""
		path = self.folder + name
		torch.save(model, path)
	
	def get_model_instance(self):
		"""

		:return:
		"""
		return self.nn_class(**self.model_hyper_param)
	
	def get_model_parameter_shapes(self):
		"""
		Return the shapes list of the network parameters
		:return: the list of the shapes
		"""
		model = self.get_model_instance()
		return {name: param.shape for name, param in model.state_dict().items()}
	
	@staticmethod
	def normal_sample_using_shapes_list(shapes, mu: float = 0, sigma: float = 1):
		"""

		:param shapes:
		:param mu:
		:param sigma:
		:return:
		"""
		return tuple(torch.randn(shape) * sigma + mu for _, shape in shapes.items())
	
	def init_individual(self):
		"""

		:return:
		"""
		means = self.normal_sample_using_shapes_list(self.parameter_shapes)
		sigmas = self.normal_sample_using_shapes_list(self.parameter_shapes, 2, 0)
		return means, sigmas
	
	def run_sampling(self, individual):
		# sample parameters
		means, sigmas = individual
		samples = []
		for mean, sigma in zip(means, sigmas):
			std = sigma ** 0.5
			sample = [torch.randn_like(mean) * std + mean for _ in range(self.sampling_size)]
			samples.append(sample)
		
		ret = list(zip(*samples))
		return ret
	
	def evaluate_individual(self, model_param, frames=20000, env_seed=None):
		"""计算个体在给定batch上的fitness值
		"""
		if env_seed is None:
			env_seed = random.randint(0, 2147483647)
		model = self.get_model_instance()
		param_dict = {name: param for name, param in zip(self.parameter_shapes, model_param)}
		model.load_state_dict(param_dict)
		g_reward = 0
		# self.env.seed(env_seed)
		state, info = self.env.reset(seed=env_seed)
		images = []
		images.append(state.__array__()[0, :, :])
		for _ in range(frames):
			obs = torch.from_numpy(state.__array__()[None] / 255).float()
			action_prob = model(obs)
			action = torch.argmax(action_prob)
			state, reward, terminated, truncated, info = self.env.step(action)
			images.append(state.__array__()[0, :, :])
			done = terminated or truncated
			g_reward += reward
			if done:
				break
		return g_reward, images
	
	def train(self):
		"""开始训练
		"""
		individual = self.population[0]
		samples = self.run_sampling(individual)
		sample = samples[0]
		
		env_seed = 0
		step = 0
		buffer = deque(maxlen=self.buffer_size)
		
		self.env.action_space.seed(env_seed)
		self.env.reset(seed=env_seed)
		print('start init buffer')
		s = time.time()
		while len(buffer) < self.buffer_size:
			step+=1
			action = self.env.action_space.sample()
			state, reward, terminated, truncated, info = self.env.step(action)
			if 0.2 > random.random():
				buffer.append(torch.tensor(state).to(nn_utils.get_device()).float() / 255)
			done = terminated or truncated
			if done:
				env_seed += 1
				self.env.action_space.seed(env_seed)
				self.env.reset(seed=env_seed)
		print(f'init buffer: {time.time()-s} seconds, {step} steps.\n{(time.time()-s)/step*25000} seconds per 25k actions.\n')
		
		print('start init test')
		for step in range(self.iteration):
			s = time.time()
			
			load_model_params(sample, buffer, **self.rollout_params)
			print(time.time()-s)
		print("+++++=====Training ended=====+++++")
	
	def final(self):
		# self.log_training_setting()
		best_solutions = ray.get([search_worker.draw.remote() for search_worker in self.search_workers])
		for i, solution in enumerate(best_solutions):
			score, frames = self.evaluate_individual(solution)
			display_frames_as_gif(frames, f"{i}_score_{score}")


if __name__ == '__main__':
	# 参数
	parser = argparse.ArgumentParser(description="A-NCNES")
	
	# Train
	parser.add_argument('--time_budget', default=300, type=int, help='Time budget in minutes')
	parser.add_argument('--iterations', default=1000, type=int, help='Number of iterations')
	parser.add_argument('--enable_time_limit', default=False, type=bool, help='Enable time limit')
	parser.add_argument('--population_size', default=1, type=int, help='Population size')
	parser.add_argument('--sampling_size', default=1, type=int, help='Sampling size for each individual')
	parser.add_argument('--buffer_size', default=500, type=int, help='Buffer_size')
	parser.add_argument('--buffer_updating_rate', default=0.25, type=float, help='The rate to update buffer each step')
	parser.add_argument('--frame_limit', default=25000, type=int, help='Frame limit for each rollout')
	
	# Module
	parser.add_argument('--nn_class', default="DQN_Atari", type=str, help='The network model')
	parser.add_argument('--input_channels', default=4, type=int, help='The number of frames to input')
	
	# Environment
	parser.add_argument('--env_name', default="Enduro", type=str, help='The network model')
	
	params = parser.parse_args()
	
	task_name = f"rollout_time_test_enduro_{params.population_size}_{params.sampling_size}"  # input("task_name:")
	os.makedirs(f"/results/A-NCNES/{task_name}/", exist_ok=True)
	os.chdir(f"/results/A-NCNES/{task_name}/")
	print(os.getcwd())
	os.makedirs(f"logs/", exist_ok=True)
	torch.manual_seed(0)
	
	# ray.init()
	trainer = ANESTrainer(params)
	
	try:
		trainer.train()
	except Exception as e:
		print(e)
