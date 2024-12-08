import argparse
import copy
import json
import logging
import math
import os
import random
import time

import ray
from ray.util.actor_pool import ActorPool
import torch

from src.env.utils import get_env
from src.nn import nn_utils
from src.plt_utils import *
from utils import RemoteRolloutWorker

AGENT_NAME = "PNES"
RESULT_ROOT_DIR = f"/results/{AGENT_NAME}"
os.makedirs(RESULT_ROOT_DIR, exist_ok=True)


@ray.remote(num_cpus=1)
class RolloutWorker:
	def __init__(self, nn_model, model_param, env_name, frame_limit, env_param=None):
		self.model = nn_model(**model_param)
		self.model.set_parameter_grad(grad=False)
		self.parameter_scale = self.model.get_parameters_amount()
		
		self.env = get_env(env_name, env_param)
		self.env_seed = random.randint(0, 2147483647)
		
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
				step += 1
				if terminated or truncated:
					break
		return g_reward, step


class NESSearch:
	def __init__(self, worker_id, random_seed, nn_model, model_param, parameter_scale, forward_workers, folder,
	             sampling_size=15, phi=0.0001, init_eta=(0.5, 0.1)):
		
		self.temp_iteration_res = None
		self.avg_fits = []
		self.best_fits = []
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
		
		self.folder = folder
		os.chdir(folder)
		self.logger = self.init_logger()
	
	def init_logger(self):
		logger = logging.getLogger(f"NESWorker #{self.worker_id}")
		logger.setLevel(level=logging.DEBUG)
		
		formatter = logging.Formatter('%(levelname)s - %(asctime)s: %(message)s')
		file_handler = logging.FileHandler(f'./logs/worker{self.worker_id}.log')
		file_handler.setLevel(level=logging.INFO)
		file_handler.setFormatter(formatter)
		
		stream_handler = logging.StreamHandler()
		stream_handler.setLevel(logging.DEBUG)
		stream_handler.setFormatter(formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(stream_handler)
		return logger
	
	def draw(self, folder):
		os.chdir(folder)
		# torch.save(self.best_solution[0], f"./Worker{self.worker_id}_best_solution_{self.best_solution[1]}.pt")
		x = list(range(1, len(self.best_fits) + 1))
		
		saving_pic_multi_line(x, (self.best_fits, self.avg_fits), f'Training Fitness with init_eta: {self.init_eta}',
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
	
	@staticmethod
	def check_nan(arr, line, name):
		for i in arr:
			if torch.isnan(i).any():
				print(f"Line {line}: {name} has nan value.")
				return True
		return False
	
	def get_best_solution(self):
		return self.best_solution[0]
	
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
		new_sigma = torch.clamp(sigma + eta[1] * (f_s ** -1) * d_f_s, 1e-6, 1e2)
		
		self.individual = (new_mean, new_sigma)
	
	def search(self, progress):
		self.run_sampling()
		
		futures = self.forward_pool.map(lambda actor, v: actor.rollout.remote(*v),
		                                [(sample,) for sample in self.samples])
		
		fits = []
		steps = 0
		for fit, step in futures:
			fits.append(fit)
			steps += step
		pairs = [list(row) for row in zip(self.samples, fits)]
		pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
		
		step_best_solution = copy.copy(pairs[0])
		if (self.best_solution is None) or step_best_solution[1] > self.best_solution[1]:
			self.best_solution = step_best_solution
		avg_fit = sum([i[1] for i in pairs]) / self.sampling_size
		# self.logger.info(f"best_fit: {step_best_solution[1]}, avg_fit: {avg_fit}")
		self.best_fits.append(step_best_solution[1])
		self.avg_fits.append(avg_fit)
		self.update(self.calculate_delta(pairs), self.cal_etas(progress))
		self.step += 1
		return step_best_solution[1], steps


@ray.remote
class NESWorker(NESSearch):
	pass


class PNESTrainer:
	def __init__(self, args):
		"""
		"""
		self.args = args
		# Modules
		self.nn_class = nn_utils.get_network_class(args.nn_class)
		# Training
		self.total_frames = args.total_frames
		self.population_size = args.population_size  # population_size
		self.sampling_size = args.sampling_size
		self.best_size = 5
		self.time_budget = args.time_budget * 60
		self.enable_time_limit = args.enable_time_limit
		self.frame_limit = args.frame_limit
		
		# hyperparameters
		self.lr_mean = args.lr_mean
		self.lr_sigma = args.lr_sigma
		
		# check env
		self.env_name = args.env_name
		self.env = get_env(self.env_name)
		
		# test
		self.test_episodes = args.test_episodes
		
		self.model_hyper_param = {"in_channels": args.input_channels,
		                          "num_actions": int(self.env.action_space.n)}
		self.search_hyper_param = {'sampling_size': self.sampling_size,
		                           'init_eta': (self.lr_mean, self.lr_sigma)}
		
		# init normalized fitness by ranking
		self.normalized_fitness_list = [max(math.log(self.sampling_size / 2 + 1) - math.log(i + 1), 0)
		                                for i in range(self.sampling_size)]
		temp_sum = sum(self.normalized_fitness_list)
		self.normalized_fitness_list = [i / temp_sum - 1 / self.sampling_size for i in self.normalized_fitness_list]
		
		# init distribution population
		self.parameter_scale = self.get_parameter_scale()
		
		self.search_workers = []
		self.best_solutions = []
		self.best_training_fitness = []
		self.best_test_res = []
		self.testing_forward_pool = ActorPool([RemoteRolloutWorker.remote(
			self.nn_class, self.model_hyper_param, self.env_name, self.frame_limit, envs_num=self.test_episodes), ])
		self.new_testing_worker = 1
		
		self.folder = os.getcwd()
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
	
	def test_episode(self, model, env_seed=None):
		if env_seed is None:
			env_seed = random.randint(0, 2147483647)
		
		g_reward = 0
		# self.env.seed(env_seed)
		state, info = self.env.reset(seed=env_seed)
		images = [self.env.render()]
		# images.append(state.__array__()[0, :, :])
		while True:
			obs = torch.from_numpy(state.__array__()[None] / 255).float()
			action_prob = model(obs)
			action = torch.argmax(action_prob)
			state, reward, terminated, truncated, info = self.env.step(action)
			images.append(self.env.render())
			done = terminated or truncated
			g_reward += reward
			if done:
				break
		return g_reward, images
	
	def test_individual(self, model_param, env_seed=None):
		scores = []
		best_episode = None
		model = self.get_model_instance()
		model.set_parameters(model_param)
		for i in range(self.test_episodes):
			res = self.test_episode(model, env_seed)
			scores.append(res[0])
			if best_episode is None or best_episode[0] > res[0]:
				best_episode = res
		return best_episode, scores
	
	def get_best_solution(self, best_samples_fits):
		step_best_training_fit, step_best_index = max((fit, i) for i, fit in enumerate(best_samples_fits))
		self.best_training_fitness.append(step_best_training_fit)
		return ray.get(self.search_workers[step_best_index].get_best_solution.remote())
	
	def training_test(self, solution, step_time):
		self.testing_forward_pool.submit(lambda a, v: a.rollout.remote(v), solution)
		if self.new_testing_worker > 0:
			self.testing_forward_pool.push(RemoteRolloutWorker.remote(
				self.nn_class, self.model_hyper_param, self.env_name, self.frame_limit, self.test_episodes))
			self.new_testing_worker -= 1
		else:
			s_t = time.time()
			self.best_test_res.append(self.testing_forward_pool.get_next())
			waiting_time = time.time() - s_t
			if waiting_time > 3.0:
				self.new_testing_worker += (int(waiting_time / step_time) + 1)
				print(f'Waiting time: {waiting_time}, adding {(int(waiting_time / step_time) + 1)} worker')
	
	def get_training_test(self):
		while self.testing_forward_pool.has_next():
			self.best_test_res.append(self.testing_forward_pool.get_next())
			
	def init_training(self):
		worker_seeds = torch.randint(0, 2147483648, (self.population_size,)).tolist()
		forward_workers = []
		for i, seed in enumerate(worker_seeds):
			forward_worker = [
				RolloutWorker.remote(self.nn_class, self.model_hyper_param, self.env_name,
				                     self.frame_limit) for _ in
				range(self.sampling_size)]
			forward_workers.extend(forward_worker)
			# noinspection PyArgumentList
			actor = NESWorker.remote(i, seed, self.nn_class, self.model_hyper_param, self.parameter_scale,
			                         forward_worker, self.folder,
			                         **self.search_hyper_param)
			self.search_workers.append(actor)
	
	def train_step(self, progress):
		search_tasks = []
		step_frames = 0
		for search_worker in self.search_workers:
			search_tasks.append(search_worker.search.remote(progress))
		# print(search_tasks) # diff
		
		search_res = ray.get(search_tasks)
		best_samples_fits = []
		for fit, frames in search_res:
			best_samples_fits.append(fit)
			step_frames += frames
		
		step_best_solution = self.get_best_solution(best_samples_fits)
		return step_best_solution, best_samples_fits, step_frames
	
	def train(self):
		"""开始训练
		"""
		self.init_training()
		
		start_time = time.time()
		cost_frames = 0
		print("+++++=====Training start=====+++++")
		while cost_frames < self.total_frames:
			s = time.time()
			if self.enable_time_limit:
				t = s - start_time
				if t >= self.time_budget:
					break
				progress = max(t / self.time_budget, cost_frames / self.total_frames)
			else:
				progress = cost_frames / self.total_frames
			
			step_best_solution, best_samples_fits, step_frames = self.train_step(progress)
			cost_frames += step_frames
			
			step_time = time.time() - s
			print(f"\n==={cost_frames}/{self.total_frames}=== {step_time}s")
			print(best_samples_fits)
			
			# test step_best_solution
			self.training_test(step_best_solution, step_time)
		
		self.get_training_test()
		print("+++++=====Training ended=====+++++")
	
	# def log_training_setting(self):
	# 	with open("./training_setting",'r') as f:
	
	def final(self):
		
		ray.get([search_worker.draw.remote(self.folder) for search_worker in self.search_workers])
		
		# best_s = 0
		# best_f = None
		# for i, solution in enumerate(best_solutions):
		# 	for j in range(20):
		# 		score, frames = self.test_individual(solution)
		# 		if score >= best_s:
		# 			best_s = score
		# 			best_f = frames
		# if best_f is not None:
		# 		display_frames_as_gif(best_f, f"best_final_test_score_{best_s}")
		x = list(range(1, len(self.best_training_fitness) + 1))
		average_test_score, average_test_steps = tuple(zip(*self.best_test_res))
		saving_pic_multi_line(x, (self.best_training_fitness, average_test_score), f'Training Fitness',
		                      f'fitness', ['Training Fitness', 'Average Testing Result'],
		                      'Score')
		training_res = {"training_fitness": self.best_training_fitness,
		                "average_test_score": average_test_score,
		                "average_test_steps": average_test_steps}
		with open('config_result.json', 'w') as f:
			json.dump(self.args.__dict__ | training_res, f)
		return training_res


def get_parser():
	arg_parser = argparse.ArgumentParser(description=AGENT_NAME)
	
	# Train
	arg_parser.add_argument('--time_budget', default=300, type=int, help='Time budget in minutes')
	arg_parser.add_argument('--total_frames', default=25000000, type=int, help='Number of total frames limit')
	arg_parser.add_argument('--enable_time_limit', default=False, type=bool, help='Enable time limit')
	arg_parser.add_argument('--population_size', default=5, type=int, help='Population size')
	arg_parser.add_argument('--sampling_size', default=15, type=int, help='Sampling size for each individual')
	arg_parser.add_argument('--frame_limit', default=10000, type=int, help='Frame limit for each rollout')
	
	arg_parser.add_argument('--lr_mean', default=0.2, type=float, help='Initial value of mean')
	arg_parser.add_argument('--lr_sigma', default=0.1, type=float, help='Initial value of sigma')
	
	# Module
	arg_parser.add_argument('--nn_class', default="DQN_Atari", type=str, help='The network model')
	arg_parser.add_argument('--input_channels', default=4, type=int, help='The number of frames to input')
	
	# Environment
	arg_parser.add_argument('--env_name', default="Freeway", type=str, help='The network model')
	
	# Test
	arg_parser.add_argument('--test_episodes', default=15, type=int, help='The number of episodes in testing')
	
	return arg_parser


def train_once(args, task_name=None):
	if task_name is None:
		task_name = f"{args.env_name}"  # input("task_name:")
	os.makedirs(f"./{task_name}/", exist_ok=False)
	os.chdir(f"./{task_name}/")
	print(os.getcwd())
	os.makedirs(f"logs/", exist_ok=True)
	
	# torch.manual_seed(0)
	trainer = PNESTrainer(args)
	
	try:
		trainer.train()
	except Exception as e:
		print(e)
	finally:
		res = trainer.final()
	
	os.chdir('../')
	return res


def train_groups(group_args, num_tasks=10, task_group_name=None):
	if task_group_name is None:
		task_group_name = f"{num_tasks}-{group_args.env_name}"
	os.makedirs(f"./{task_group_name}/", exist_ok=False)
	os.chdir(f"./{task_group_name}/")
	ress = []
	for i in range(num_tasks):
		ress.append(train_once(group_args, f'{i}'))
	training_fitness = []
	average_test_score = []
	iterations = None
	for res in ress:
		training_fitness.append(res['training_fitness'])
		average_test_score.append(res['average_test_score'])
		if iterations is None:
			iterations = len(res['training_fitness'])
		else:
			iterations = max(iterations, len(res['training_fitness']))
	training_fitness = [arr[:iterations] for arr in training_fitness]
	average_test_score = [arr[:iterations] for arr in average_test_score]
	show_multi_line_and_area(list(range(1, iterations + 1)), (training_fitness, average_test_score),
	                         f"{AGENT_NAME}: {group_args.env_name} training result with {num_tasks} runs",
	                         "sum_result", ['Training Fitness', 'Average Test Score'],
	                         "Score")
	os.chdir(f"../")


if __name__ == '__main__':
	os.chdir(RESULT_ROOT_DIR)
	parser = get_parser()
	run_args = parser.parse_args()
	
	# train_once(run_args)
	train_groups(run_args, 10)
# trainer.test_best()
