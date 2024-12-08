import argparse
import copy
import logging
import math
import random
import time
from collections import deque

import ray
from ray.util.actor_pool import ActorPool
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.agent.disc_pnes import PNESTrainer
from src.env.utils import get_env
from src.nn import nn_utils
from src.plt_utils import *

AGENT_NAME = "A-NCNES"


@ray.remote(num_cpus=1)
class RolloutWorker:
	def __init__(self, diversity_worker, nn_model, model_param, env_name, frame_limit, buffer_prob, env_param=None):
		self.diversity_worker = diversity_worker
		self.model = nn_model(**model_param)
		self.model.set_parameter_grad(grad=False)
		self.parameter_scale = self.model.get_parameters_amount()
		
		self.env = get_env(env_name, env_param)
		self.env_seed = 0
		
		self.frame_limit = frame_limit
		self.buffer_prob = buffer_prob
	
	def rollout(self, model_param):
		with torch.no_grad():
			device = nn_utils.get_device()
			self.model.set_parameters(model_param)
			g_reward = 0
			step = 0
			frames = []
			state, info = self.env.reset(seed=self.env_seed)
			self.env_seed += 1
			while step < self.frame_limit:
				obs = torch.tensor(state).to(device).float() / 255
				if random.random() < self.buffer_prob:
					frames.append(obs)
				action_prob = self.model(obs.unsqueeze(0))
				action = torch.argmax(action_prob)
				state, reward, terminated, truncated, info = self.env.step(action)
				g_reward += reward
				step += 1
				if terminated or truncated:
					break
			frame_fut = ray.put(frames)
		sum_action_prob = self.diversity_sampling()
		return g_reward, frame_fut, sum_action_prob, step
	
	def diversity_sampling(self):
		with torch.no_grad():
			buffer = ray.get(ray.get(self.diversity_worker.get_buffer_fut.remote()))
			dataset = TensorDataset(torch.stack(tuple(buffer), dim=0))
			loader = DataLoader(dataset, 500, False)
			sum_action_prob = None
			for batch in loader:
				action_prob = self.model(batch[0])
				action_prob = torch.sum(action_prob, dim=[0, ])
				if sum_action_prob is None:
					sum_action_prob = action_prob
				else:
					sum_action_prob = sum_action_prob + action_prob
		return sum_action_prob / len(buffer)


@ray.remote
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
		self.update_rate = 0.5
		
		while len(self.buffer) < self.buffer_size:
			action = self.env.action_space.sample()
			state, reward, terminated, truncated, info = self.env.step(action)
			self.buffer.append(torch.tensor(state).to(nn_utils.get_device()).float() / 255)
			done = terminated or truncated
			if done:
				# self.env.seed(self.env_seed)
				self.env.action_space.seed(self.env_seed)
				self.env_seed += 1
				self.env.reset(seed=self.env_seed)
		
		self.buffer_fut = ray.put(self.buffer)
	
	def update_buffer(self, frame_futures):
		for fut in frame_futures:
			self.buffer.extend(ray.get(fut))
		random.shuffle(self.buffer)
		self.buffer_fut = ray.put(self.buffer)
	
	def get_buffer_fut(self):
		return self.buffer_fut


class NESSearch:
	def __init__(self, worker_id, random_seed, nn_model, model_param, parameter_scale, forward_workers,
	             diversity_worker, folder,
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
		self.diversity_worker = diversity_worker
		
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
	
	def diversity_sampling(self, model_param, loader):
		self.model.set_parameters(model_param)
		with torch.no_grad():
			sum_action_prob = None
			for batch in loader:
				action_prob = self.model(batch[0])
				action_prob = torch.sum(action_prob, dim=[0, ])
				if sum_action_prob is None:
					sum_action_prob = action_prob
				else:
					sum_action_prob = sum_action_prob + action_prob
		return sum_action_prob / len(loader.dataset)
	
	def get_action_probs(self):
		buffer = ray.get(ray.get(self.diversity_worker.get_buffer_fut.remote()))
		dataset = TensorDataset(torch.stack(tuple(buffer), dim=0))
		loader = DataLoader(dataset, 500, False)
		return [self.diversity_sampling(sample, loader) for sample in self.samples]
	
	@staticmethod
	def check_nan(arr, line, name):
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
		diversity_m = []
		diversity_s = []
		
		for sample, normalized_fitness, action_prob in sorted_pairs:
			diff = (sample - mean)
			temp = inverse_sigma * diff
			l_m = temp * normalized_fitness
			l_dm = action_prob * temp.unsqueeze(-1)
			l_fm = temp ** 2
			temp = l_fm - inverse_sigma
			l_s = temp * normalized_fitness
			l_ds = action_prob * temp.unsqueeze(-1)
			l_fs = temp ** 2
			
			sum_m.append(l_m)
			sum_s.append(l_s)
			sum_fm.append(l_fm)
			sum_fs.append(l_fs)
			diversity_m.append(l_dm)
			diversity_s.append(l_ds)
		
		d_f_m = torch.sum(torch.stack(sum_m), dim=0) / self.sampling_size  # 14
		d_f_s = torch.sum(torch.stack(sum_s), dim=0) / (2 * self.sampling_size)  # 15
		f_m = torch.sum(torch.stack(sum_fm), dim=0) / self.sampling_size
		f_s = torch.sum(torch.stack(sum_fs), dim=0) / (4 * self.sampling_size)
		sum_dm = torch.sum(torch.stack(diversity_m), dim=0) / self.sampling_size
		sum_ds = torch.sum(torch.stack(diversity_s), dim=0) / (2 * self.sampling_size)
		
		return d_f_m, d_f_s, f_m, f_s, sum_dm, sum_ds
	
	def cal_diversity_delta(self, diversity, action_probs):
		sum_dm, sum_ds = diversity
		
		log_2_self_action_prob = torch.log2(self.sum_action_prob + 1)
		action_prob_res = log_2_self_action_prob * len(action_probs) + len(action_probs)
		for action_prob in action_probs:
			action_prob_res = action_prob_res - torch.log2(action_prob + 1)
		
		d_d_m = torch.sum(sum_dm * action_prob_res, dim=-1)
		d_d_s = torch.sum(sum_ds * action_prob_res, dim=-1)
		
		return d_d_m, d_d_s
	
	def update(self, res_f, res_d, eta):
		d_f_m, d_f_s, f_m, f_s = res_f
		d_d_m, d_d_s = res_d
		
		mean, sigma = self.individual
		new_mean = mean + eta[0] * (f_m ** -1) * (d_f_m + self.phi * d_d_m)
		new_sigma = torch.clamp(sigma + eta[1] * (f_s ** -1) * (d_f_s + self.phi * d_d_s), 1e-6, 1e2)
		
		self.individual = (new_mean, new_sigma)
	
	def search1(self):
		self.run_sampling()
		
		futures = self.forward_pool.map(lambda actor, v: actor.rollout.remote(*v),
		                                [(sample,) for sample in self.samples])
		
		fits = []
		frames_fut = []
		action_probs = []
		steps = 0
		for fit, frame_fut, action_prob, step in futures:
			fits.append(fit)
			frames_fut.append(frame_fut)
			action_probs.append(action_prob)
			steps += step
		pairs = [list(row) for row in zip(self.samples, fits, action_probs)]
		pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
		
		step_best_solution = copy.copy(pairs[0])
		if (self.best_solution is None) or step_best_solution[1] > self.best_solution[1]:
			self.best_solution = step_best_solution
		avg_fit = sum([i[1] for i in pairs]) / self.sampling_size
		# self.logger.info(f"best_fit: {step_best_solution[1]}, avg_fit: {avg_fit}")
		self.best_fits.append(step_best_solution[1])
		self.avg_fits.append(avg_fit)
		
		self.temp_iteration_res = self.calculate_delta(pairs)
		self.sum_action_prob = torch.sum(torch.stack(action_probs), dim=0)
		return frames_fut, self.sum_action_prob, step_best_solution[1], steps
	
	def search2(self, progress, action_probs):
		
		res_d = self.cal_diversity_delta(self.temp_iteration_res[-2:], action_probs)
		# self.logger.debug(res_f)
		# self.logger.debug(res_d)
		self.update(self.temp_iteration_res[:4], res_d, self.cal_etas(progress))
		
		# self.logger.debug(f"{self.worker_id}: {self.best_solutions}")
		self.step += 1
		return


@ray.remote
class NESWorker(NESSearch):
	pass


class ANCNESTrainer(PNESTrainer):
	def __init__(self, args):
		"""
		"""
		super().__init__(args)
		# Training
		self.buffer_size = args.buffer_size
		self.buffer_updating_rate = args.buffer_updating_rate
		self.buffer_prob = self.buffer_size * self.buffer_updating_rate / (
				self.sampling_size * self.population_size * self.frame_limit)
		
		# hyperparameters
		self.phi = args.phi
		self.diversity_worker_hyper_param = {
			'nn_model': self.nn_class,
			'model_param': self.model_hyper_param,
			'env_name': self.env_name,
			'frame_limit': self.frame_limit,
			'buffer_prob': self.buffer_prob
		}
		
		self.diversity_worker = None
		
	def init_training(self):
		# noinspection PyArgumentList
		self.diversity_worker = DiversityWorker.remote(self.env_name, self.buffer_size)
		worker_seeds = torch.randint(0, 2147483648, (self.population_size,)).tolist()
		for i, seed in enumerate(worker_seeds):
			forward_worker = [
				RolloutWorker.remote(self.diversity_worker, **self.diversity_worker_hyper_param) for _ in
				range(self.sampling_size)]
			# noinspection PyArgumentList
			actor = NESWorker.remote(i, seed, self.nn_class, self.model_hyper_param, self.parameter_scale,
			                         forward_worker, self.diversity_worker, self.folder,
			                         **self.search_hyper_param)
			self.search_workers.append(actor)
	
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
			
			search_tasks1 = []
			
			for search_worker in self.search_workers:
				search_tasks1.append(search_worker.search1.remote())
			# print(search_tasks) # diff
			
			search_res1 = ray.get(search_tasks1)  # 同步
			
			frame_futures = []
			action_probs = []
			best_samples_fits = []
			for frame_fut, action_prob, fit, frames in search_res1:
				frame_futures.extend(frame_fut)
				action_probs.append(action_prob)
				best_samples_fits.append(fit)
				cost_frames += frames
			assert len(best_samples_fits) == self.population_size
			
			update_buffer_task = self.diversity_worker.update_buffer.remote(frame_futures)
			search_tasks2 = [search_worker.search2.remote(progress, action_probs) for search_worker in
			                 self.search_workers]
			# print(search_tasks) # diff
			ray.get(update_buffer_task)
			ray.get(search_tasks2)  # 同步
			
			step_best_training_fit, step_best_index = max((fit, i) for i, fit in enumerate(best_samples_fits))
			step_best_solution = ray.get(self.search_workers[step_best_index].get_best_solution.remote())
			# TODO: test step_best_solution
			
			print(f"\n==={cost_frames}/{self.total_frames}=== {time.time() - s}s")
			print(best_samples_fits)
		print("+++++=====Training ended=====+++++")


def main(args):
	task_name = f"{args.env_name}"  # input("task_name:")
	
	os.makedirs(f"/results/{AGENT_NAME}/{task_name}/", exist_ok=True)
	os.chdir(f"/results/{AGENT_NAME}/{task_name}/")
	print(os.getcwd())
	os.makedirs(f"logs/", exist_ok=True)
	torch.manual_seed(0)
	
	trainer = ANCNESTrainer(args)
	
	try:
		trainer.train()
	except Exception as e:
		print(e)
	finally:
		trainer.final()


if __name__ == '__main__':
	# 参数
	parser = argparse.ArgumentParser(description=AGENT_NAME)
	
	# Train
	parser.add_argument('--time_budget', default=300, type=int, help='Time budget in minutes')
	parser.add_argument('--total_frames', default=25000000, type=int, help='Number of total frames limit')
	parser.add_argument('--enable_time_limit', default=False, type=bool, help='Enable time limit')
	parser.add_argument('--population_size', default=5, type=int, help='Population size')
	parser.add_argument('--sampling_size', default=15, type=int, help='Sampling size for each individual')
	parser.add_argument('--buffer_size', default=10000, type=int, help='Buffer_size')
	parser.add_argument('--buffer_updating_rate', default=0.4, type=float, help='The rate to update buffer each step')
	parser.add_argument('--frame_limit', default=10000, type=int, help='Frame limit for each rollout')
	
	parser.add_argument('--lr_mean', default=0.2, type=float, help='Initial value of mean')
	parser.add_argument('--lr_sigma', default=0.1, type=float, help='Initial value of sigma')
	parser.add_argument('--phi', default=0.001, type=float, help='Initial value of the tradeoff between F and D')
	# Network
	parser.add_argument('--nn_class', default="DQN_Atari", type=str, help='The network model')
	parser.add_argument('--input_channels', default=4, type=int, help='The number of frames to input')
	
	# Environment
	parser.add_argument('--env_name', default="Breakout", type=str, help='The network model')
	
	# for Test
	# parser.add_argument('--train_size', default=2000, type=int, help='Training data size')
	# parser.add_argument('--val_size', default=500, type=int, help='Validation data size')
	# parser.add_argument('--test_size', default=500, type=int, help='Test data size')
	# parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
	ray.init()
	run_args = parser.parse_args()
	
	main(run_args)

# trainer.test_best()
