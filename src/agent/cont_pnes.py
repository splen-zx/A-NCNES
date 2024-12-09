import argparse
import copy
import logging
import os
import math
import random
import time

import ray
from ray.util.actor_pool import ActorPool
import torch
from tqdm import tqdm

from src.env.utils import get_env
from src.nn import nn_utils
from src.plt_utils import *


@ray.remote(num_cpus=1)
class RolloutWorker:
	def __init__(self, nn_model, model_param, env_name, frame_limit, env_param=None):
		self.model = nn_model(**model_param)
		self.model.set_parameter_grad(grad=False)
		self.parameter_scale = self.model.get_parameters_amount()
		
		self.env = get_env(env_name, env_param)
		self.env_seed = 0
		
		self.frame_limit = frame_limit
	
	def rollout(self, model_param):
		with torch.no_grad():
			device = nn_utils.get_device()
			self.model.set_parameters(model_param)
			g_reward = 0
			step = 0
			state, info = self.env.reset(seed=self.env_seed)
			self.env_seed += 1
			while step < self.frame_limit:
				obs = torch.tensor(state).to(device).float() / 255
				action_prob = self.model(obs.unsqueeze(0))
				action = torch.argmax(action_prob)
				state, reward, terminated, truncated, info = self.env.step(action)
				g_reward += reward
				step +=1
				if terminated or truncated:
					break
		return g_reward, step

@ray.remote
class NESWorker:
	def __init__(self, worker_id, random_seed, nn_model, model_param, parameter_scale, forward_workers,
	             sampling_size=15, phi=0.0001, init_eta=(0.5, 0.1)):
		
		self.temp_iteration_res = None
		self.avgfits = []
		self.bestfits = []
		self.sampling_size = sampling_size
		self.init_eta = init_eta
		self.phi = phi
		
		self.worker_id = worker_id
		self.random_seed = random_seed
		self.parameter_scale = parameter_scale
		self.forward_pool = ray.util.ActorPool(forward_workers)
		
		torch.manual_seed(self.random_seed)
		self.individual = self.init_individual()
		self.best_solution = None
		self.res_f = None
		self.res_self_d = None
		self.sum_action_prob = None
		self.samples = None
		self.step = 0
		
		self.model = nn_model(**model_param)
		self.model.set_parameter_grad(grad=False)
		
		# init normalized fitness by ranking
		temp = [max(math.log(self.sampling_size / 2 + 1) - math.log(i + 1), 0) for i in range(self.sampling_size)]
		temp_sum = sum(temp)
		self.normalized_fitness = [i / temp_sum - 1 / self.sampling_size for i in temp]
		self.cal_etas = lambda progress: tuple(
			eta * (math.e - math.e ** progress) / (math.e - 1) for eta in self.init_eta)
		
		self.logger = self.init_logger()
		
	
	def init_logger(self):
		logger = logging.getLogger(f"NESWorker #{self.worker_id}")
		logger.setLevel(level=logging.DEBUG)
		
		formatter = logging.Formatter('%(levelname)s - %(asctime)s: %(message)s')
		file_handler = logging.FileHandler(f'logs/worker{self.worker_id}.log')
		file_handler.setLevel(level=logging.INFO)
		file_handler.setFormatter(formatter)
		
		stream_handler = logging.StreamHandler()
		stream_handler.setLevel(logging.DEBUG)
		stream_handler.setFormatter(formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(stream_handler)
		return logger
	
	def draw(self):
		torch.save(self.best_solution[0], f"./Worker{self.worker_id}_best_solution_{self.best_solution[1]}.pt")
		x = list(range(1, len(self.bestfits) + 1))
		
		saving_pic_multi_line(x, (self.bestfits, self.avgfits), f'Training Fitness with init_eta: {self.init_eta}',
		                           f'Worker{self.worker_id}_fitness', ['best fitness', 'average fitness'],
		                           'Fitness')
		# return best solution
		
		# show_line_and_area([0,]+x, list(zip(*self.log_infos["means"])), "Means change of some params", f"Worker{self.worker_id}_Means_fig_1")
		# show_line_and_area([0,]+x, list(zip(*self.log_infos["sigmas"])), "Sigmas change of some params", f"Worker{self.worker_id}_Sigmas_fig_1")
		return self.best_solution[0]
	
	def init_individual(self):
		mean = torch.randn(self.parameter_scale)
		sigma = torch.randn(self.parameter_scale) * 0 + 2
		return mean, sigma
	
	def run_sampling(self):
		# sample parameters
		mean, sigma = self.individual
		std = sigma ** 0.5
		self.samples = [torch.randn(self.parameter_scale) * std + mean for _ in range(self.sampling_size)]
	
	
	def check_nan(self, arr, line, name):
		for i in arr:
			if torch.isnan(i).any():
				print(f"Line {line}: {name} has nan value.")
				return True
		return False
	
	def calculate_delta(self, sorted_pairs):
		mean, sigma = self.individual
		for i, pair in enumerate(sorted_pairs):
			pair[1] = self.normalized_fitness[i]
		inverse_sigma = sigma ** -1
		
		# cal d_fitness and fisher
		sum_m = []
		sum_s = []
		sum_fm = []
		sum_fs = []

		for sample, normalized_fitness in sorted_pairs:
			diff = (sample - mean)
			temp = inverse_sigma * diff
			l_m = temp * normalized_fitness
			l_fm = temp ** 2
			temp = l_fm - inverse_sigma
			l_s = temp * normalized_fitness
			l_fs = temp ** 2
			
			sum_m.append(l_m)
			sum_s.append(l_s)
			sum_fm.append(l_fm)
			sum_fs.append(l_fs)
		
		d_f_m = torch.sum(torch.stack(sum_m), dim=0) / self.sampling_size  # 14
		d_f_s = torch.sum(torch.stack(sum_s), dim=0) / (2 * self.sampling_size)  # 15
		f_m = torch.sum(torch.stack(sum_fm), dim=0) / self.sampling_size
		f_s = torch.sum(torch.stack(sum_fs), dim=0) / (4 * self.sampling_size)
		
		return d_f_m, d_f_s, f_m, f_s,
	
	
	def update(self, res_f, eta):
		d_f_m, d_f_s, f_m, f_s = res_f
		
		mean, sigma = self.individual
		new_mean = mean + eta[0] * (f_m ** -1) * d_f_m
		new_sigma =	torch.clamp(sigma + eta[1] * (f_s ** -1) * d_f_s, 1e-6, 1e2)
		
		self.individual = (new_mean, new_sigma)
	
	def search(self, progress):
		self.run_sampling()
		
		futs = self.forward_pool.map(lambda actor, v: actor.rollout.remote(*v),
		                                    [(sample,) for sample in self.samples])
		
		fits = []
		steps = 0
		for fit, step in futs:
			fits.append(fit)
			steps +=step
		pairs = [list(row) for row in zip(self.samples, fits)]
		pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
		
		step_best_solution = copy.copy(pairs[0])
		if (self.best_solution is None) or step_best_solution[1] > self.best_solution[1]:
			self.best_solution = step_best_solution
		avg_fit = sum([i[1] for i in pairs]) / self.sampling_size
		# self.logger.info(f'best_fit: {step_best_solution[1]}, avg_fit: {avg_fit}')
		self.bestfits.append(step_best_solution[1])
		self.avgfits.append(avg_fit)
		self.update(self.calculate_delta(pairs), self.cal_etas(progress))
		self.step += 1
		return step_best_solution[1], steps
	


class PNESTrainer:
	def __init__(self, params):
		"""
		"""
		self.params = params
		# Modules
		self.nn_class = nn_utils.get_network_class(params.nn_class)
		# Training
		self.total_frames = params.total_frames
		self.population_size = params.population_size  # population_size
		self.sampling_size = params.sampling_size
		self.best_size = 5
		self.time_budget = params.time_budget * 60
		self.enable_time_limit = params.enable_time_limit
		self.frame_limit = params.frame_limit
		
		# hyperparameters
		self.lr_mean = params.lr_mean
		self.lr_sigma = params.lr_sigma
		
		# check env
		self.env_name = params.env_name
		self.env = get_env(self.env_name)
		
		self.model_hyper_param = {"in_channels": params.input_channels,
		                          "num_actions": int(self.env.action_space.n)}
		self.search_hyper_param = {'sampling_size': self.sampling_size,
		                           'init_eta': (self.lr_mean, self.lr_sigma)}
		
		# init normalized fitness by ranking
		self.normalized_fitnesses = [max(math.log(self.sampling_size / 2 + 1) - math.log(i + 1), 0)
		                             for i in range(self.sampling_size)]
		temp_sum = sum(self.normalized_fitnesses)
		self.normalized_fitnesses = [i / temp_sum - 1 / self.sampling_size for i in self.normalized_fitnesses]
		
		# init distribution population
		self.parameter_scale = self.get_parameter_scale()
		
		self.search_workers = []
		self.best_solutions = []
		self.forward_pool = None
		print(f"total_frames: {self.total_frames}")
		print(f"population_size: {self.population_size}")
		print(f"sampling_size: {self.sampling_size}")
		print(f"time_budget: {self.time_budget}s")
		print(f"lr_mean: {self.lr_mean}")
		print(f"lr_sigma: {self.lr_sigma}")
	
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
	
	def get_parameter_scale(self):
		"""
		Return the shapes list of the network parameters
		:return: the list of the shapes
		"""
		model = self.get_model_instance()
		return model.get_parameters_amount()
	
	def init_individual(self):
		mean = torch.randn(self.parameter_scale)
		sigma = torch.randn(self.parameter_scale) * 0 + 2
		return mean, sigma
	
	def test_individual(self, model_param, env_seed=None):
		"""计算个体在给定batch上的fitness值
		"""
		if env_seed is None:
			env_seed = random.randint(0, 2147483647)
		model = self.get_model_instance()
		model.set_parameters(model_param)
		g_reward = 0
		# self.env.seed(env_seed)
		state, info = self.env.reset(seed=env_seed)
		images = []
		images.append(state.__array__()[0, :, :])
		while True:
			obs = torch.from_numpy(state.__array__()[None] / 255).float()
			action_prob = model(obs)
			action = torch.argmax(action_prob)
			state, reward, terminated, truncated, info = self.env.step(action)
			image = self.env.render()
			images.append(image)
			done = terminated or truncated
			g_reward += reward
			if done:
				break
		return g_reward, images
	
	
	def train(self):
		"""开始训练
		"""
		worker_seeds = torch.randint(0, 2147483648, (self.population_size,)).tolist()
		forward_workers = []
		for i, seed in enumerate(worker_seeds):
			forward_worker = [
				RolloutWorker.remote(self.nn_class, self.model_hyper_param, self.env_name,
				                     self.frame_limit) for _ in
				range(self.sampling_size)]
			forward_workers.extend(forward_worker)
			actor = NESWorker.remote(i, seed, self.nn_class, self.model_hyper_param, self.parameter_scale, forward_worker,
			                         **self.search_hyper_param)
			self.search_workers.append(actor)
		self.forward_pool = ActorPool(forward_workers)
		
		start_time = time.time()
		cost_frames = 0
		print("+++++=====Training start=====+++++")
		while cost_frames <self.total_frames:
			s = time.time()
			if self.enable_time_limit:
				t = s - start_time
				if t >= self.time_budget:
					break
				progress = max(t / self.time_budget, cost_frames / self.total_frames)
			else:
				progress = cost_frames / self.total_frames
			
			search_tasks = []

			for search_worker in self.search_workers:
				search_tasks.append(search_worker.search.remote(progress))
			# print(search_tasks) # diff

			search_res = ray.get(search_tasks)  # 同步
			best_samples_fits = []
			for fit, frames in search_res:
				best_samples_fits.append(fit)
				cost_frames += frames
			assert len(best_samples_fits) == self.population_size
			
			# print(search_tasks) # diff
			
			print(f"\n==={cost_frames}/{self.total_frames}=== {time.time() - s}s")
			print(best_samples_fits)
		print("+++++=====Training ended=====+++++")
	
	# def log_training_setting(self):
	# 	with open("./training_setting",'r') as f:
		
	def final(self):
		# self.log_training_setting()
		best_solutions = ray.get([search_worker.draw.remote() for search_worker in self.search_workers])
		for i, solution in enumerate(best_solutions):
			best_s = 0
			best_f = None
			for j in range(20):
				score, frames = self.test_individual(solution)
				if score > best_s:
					best_s = score
					best_f = frames
			if best_s > 0:
				display_frames_as_gif(best_f, f"{i}_score_{score}")


if __name__ == '__main__':
	agent_name = "PNES"
	# 参数
	parser = argparse.ArgumentParser(description=agent_name)
	
	# Train
	parser.add_argument('--time_budget', default=300, type=int, help='Time budget in minutes')
	parser.add_argument('--total_frames', default=25000000, type=int, help='Number of total frames limit')
	parser.add_argument('--enable_time_limit', default=False, type=bool, help='Enable time limit')
	parser.add_argument('--population_size', default=5, type=int, help='Population size')
	parser.add_argument('--sampling_size', default=15, type=int, help='Sampling size for each individual')
	parser.add_argument('--frame_limit', default=10000, type=int, help='Frame limit for each rollout')
	
	parser.add_argument('--lr_mean', default=0.2, type=float, help='Initial value of mean')
	parser.add_argument('--lr_sigma', default=0.1, type=float, help='Initial value of sigma')
	# Network
	parser.add_argument('--nn_class', default="DQN_Atari", type=str, help='The network model')
	parser.add_argument('--input_channels', default=4, type=int, help='The number of frames to input')
	
	# Environment
	parser.add_argument('--env_name', default="Freeway", type=str, help='The network model')
	
	# for Test
	# parser.add_argument('--train_size', default=2000, type=int, help='Training data size')
	# parser.add_argument('--val_size', default=500, type=int, help='Validation data size')
	# parser.add_argument('--test_size', default=500, type=int, help='Test data size')
	# parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
	
	params = parser.parse_args()
	
	task_name = f"{params.env_name}"  # input("task_name:")
	os.makedirs(f"/results/{agent_name}/{task_name}/", exist_ok=True)
	os.chdir(f"/results/{agent_name}/{task_name}/")
	print(os.getcwd())
	os.makedirs(f"logs/", exist_ok=True)
	# torch.manual_seed(0)
	
	ray.init()
	trainer = PNESTrainer(params)

	trainer.train()
	res = trainer.final()
	
# trainer.test_best()
