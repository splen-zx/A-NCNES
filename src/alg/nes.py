import copy
import logging
import os
import torch
import math
import time
import ray

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from PointerNet import PointerNet
from Data_Generator import TSPDataset
import matplotlib.pyplot as plt


def logger(log, file_path):
	with open(file_path, 'a') as f:
		f.write(log + '\n')


def custom_collate_fn(batch):
	batch = [data for data in batch if data is not None]  # 过滤掉为 None 的样本
	return torch.utils.data.dataloader.default_collate(batch)


def calculate_path_length(y, cities):
	"""计算TSP实例结果序列的的路径长度

	Args:
		y (int arr): TSP解的城市序列
		cities ([(x11,x12),...,]): 城市坐标

	Returns:
		double : 解的回路长度
	"""
	path_length = 0
	
	for i in range(len(y) - 1):
		city1 = cities[y[i]]
		city2 = cities[y[i + 1]]
		distance = math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)
		path_length += distance
	
	# 添加回到起点的路径长度
	city1 = cities[y[-1]]
	city2 = cities[y[0]]
	distance = ((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2) ** 0.5
	path_length += distance
	
	return path_length


def calculate_paths(ys, cities_list):
	total_length = 0.0
	for i in range(len(ys)):
		y = ys[i]
		cities = cities_list[i]
		total_length += calculate_path_length(y, cities)
	return total_length / len(ys)


@ray.remote
class ForwardWorker:
	def __init__(self, model_param):
		self.model = PointerNet(**model_param)
		self.parameter_shapes = {name: param.shape for name, param in self.model.state_dict().items()}
	
	def forwarding_evaluating(self, model_param, dataset):
		with torch.no_grad():
			param_dict = {name: param for name, param in zip(self.parameter_shapes, model_param)}
			self.model.load_state_dict(param_dict)
			o, p = self.model(dataset)
		res = self.calculate_paths(p, dataset)
		# if not res:
		# 	name = torch.randint(0, 100000000, (1,)).tolist()[0]
		# 	torch.save(model_param, f"error_model_{name}.pt")
		# 	torch.save(dataset, f"error_data_{name}.pt")
		# 	raise Exception("!!!")
		return res
	
	def calculate_paths(self, ys, cities_list):
		total_length = 0.0
		for i in range(len(ys)):
			y = ys[i]
			cities = cities_list[i]
			total_length += self.calculate_path_length(y, cities)
		return total_length / len(ys)
	
	def calculate_path_length(self, y, cities):
		path_length = 0
		
		for i in range(len(y) - 1):
			city1 = cities[y[i]]
			city2 = cities[y[i + 1]]
			distance = math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)
			path_length += distance
		
		# 添加回到起点的路径长度
		city1 = cities[y[-1]]
		city2 = cities[y[0]]
		distance = math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)
		path_length += distance
		
		return path_length


@ray.remote
class NESWorker:
	def __init__(self, worker_id, random_seed, parameter_shapes, forward_workers,
	             sampling_size=15, init_eta=(0.5, 0.1)):
		
		self.avgfits = []
		self.bestfits = []
		self.sampling_size = sampling_size
		self.init_eta = init_eta
		
		self.worker_id = worker_id
		self.random_seed = random_seed
		self.parameter_shapes = parameter_shapes
		self.forward_pool = ray.util.ActorPool(forward_workers)
		
		torch.manual_seed(self.random_seed)
		self.individual = self.init_individual()
		self.best_solutions = None
		
		# init normalized fitness by ranking
		temp = [max(math.log(self.sampling_size / 2 + 1) - math.log(i + 1), 0) for i in range(self.sampling_size)]
		temp_sum = sum(temp)
		self.normalized_fitness = [i / temp_sum - 1 / self.sampling_size for i in temp]
		self.cal_etas = lambda progress: tuple(
			eta * (math.e - math.e ** progress) / (math.e - 1) for eta in self.init_eta)
		
		self.logger = self.init_logger()
		self.log_infos = []
	
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
		torch.save(self.best_solutions[0], './best_solution.pt')
		x = list(range(1, len(self.log_infos) + 1))
		infos = list(zip(*self.log_infos))
		self.saving_pic_multi_line(x, (self.bestfits, self.avgfits), f'Training Fitness with init_eta: {self.init_eta}',
		                           f'Worker{self.worker_id}_fitness', ['best fitness', 'average fitness'],
		                           'Fitness')
		for i, data in enumerate(infos):
			self.saving_pic(x, data, f'Param{i}', f'Worker{self.worker_id}_param{i}', 'Param')
	
	@staticmethod
	def saving_pic(x, y, fig_name, image_name, y_label, x_label='Training Batch'):
		plt.figure(figsize=(8, 6))
		plt.plot(x, y, marker=None, linestyle='-')
		# plt.legend()
		plt.title(fig_name)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.savefig(os.path.join(f'./{image_name}.png'))
	
	@staticmethod
	def saving_pic_multi_line(x, y, fig_name, image_name, line_labels, y_label, x_label='Training Batch'):
		plt.figure(figsize=(8, 6))
		for data, name in zip(y, line_labels):
			plt.plot(x, data, marker=None, linestyle='-', label=name)
		plt.legend()
		plt.title(fig_name)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.savefig(os.path.join(f'./{image_name}.png'))
	
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
	
	def calculate_delta(self, sorted_pairs):
		mean, sigma = self.individual
		for i, pair in enumerate(sorted_pairs):
			pair[1] = self.normalized_fitness[i]
		
		# cal d_fitness and fisher
		sum_m = []
		sum_s = []
		sum_fm = []
		sum_fs = []
		inverse_sigma = [1.0 / tensor for tensor in sigma]
		for sample, normalized_fitness in sorted_pairs:
			l_m = []
			l_s = []
			l_fm = []
			l_fs = []
			for mean_tensor, inverse_sigma_tensor, sample_tensor in zip(mean, inverse_sigma, sample):
				diff = (sample_tensor - mean_tensor)
				temp = inverse_sigma_tensor * diff
				l_m.append(temp * normalized_fitness)
				temp = temp ** 2
				l_fm.append(temp)
				temp = temp - inverse_sigma_tensor
				l_s.append(temp * normalized_fitness)
				l_fs.append(temp ** 2)
			sum_m.append(l_m)
			sum_s.append(l_s)
			sum_fm.append(l_fm)
			sum_fs.append(l_fs)
		
		d_f_m = [torch.sum(torch.stack(row), dim=0) / self.sampling_size for row in zip(*sum_m)]  # 14
		d_f_s = [torch.sum(torch.stack(row), dim=0) / (2 * self.sampling_size) for row in zip(*sum_s)]  # 15
		f_m = [torch.sum(torch.stack(row), dim=0) / self.sampling_size for row in zip(*sum_fm)]
		f_s = [torch.sum(torch.stack(row), dim=0) / (4 * self.sampling_size) for row in zip(*sum_fs)]
		# self.logger.debug(f"d_f_m: {len(d_f_m)}")
		# self.logger.debug(f"d_f_s: {len(d_f_s)}")
		# self.logger.debug(f"f_m: {len(f_m)}")
		# self.logger.debug(f"f_s: {len(f_s)}")
		return d_f_m, d_f_s, f_m, f_s
	
	def update(self, res_f, eta):
		d_f_m, d_f_s, f_m, f_s = res_f
		mean, sigma = self.individual
		new_mean = tuple(
			m + eta[0] * (fisher ** -1) * f for m, fisher, f in zip(mean, f_m, d_f_m))
		new_sigma = tuple(
			torch.clamp(sigma + eta[1] * (fisher ** -1) * f, 1e-8, 1e4) for sigma, fisher, f in
			zip(sigma, f_s, d_f_s))
		self.individual = (new_mean, new_sigma)
	
	def search(self, batched_data, progress):
		# self.logger.debug(f"progress: {progress}")
		samples = self.run_sampling()
		# self.logger.debug(samples[0][0])
		
		fits = self.forward_pool.map(lambda actor, v: actor.forwarding_evaluating.remote(*v),
		                             [(sample, batched_data) for sample in samples])
		fits = list(fits)
		# self.logger.debug(f"{self.worker_id}: {fits}")
		
		pairs = [list(row) for row in zip(samples, fits)]
		pairs = sorted(pairs, key=lambda x: x[1], reverse=False)
		self.best_solutions = copy.copy(pairs[0])
		avg_fit = sum(i[1] for i in pairs) / self.sampling_size
		# self.logger.debug(pairs)
		
		res_f = self.calculate_delta(pairs)
		# self.logger.debug(res_f)
		# self.logger.debug(res_d)
		self.update(res_f, self.cal_etas(progress))
		
		# self.logger.debug(f"{self.worker_id}: {self.best_solutions}")
		self.logger.info(f'best_fit: {self.best_solutions[1]}, avg_fit: {avg_fit}')
		self.log_infos.append([self.individual[0][1][0, 0].item(), self.individual[0][4][0, 0].item(),
		                       self.individual[0][23][0, 0].item(), self.individual[0][30][0, 0].item(),
		                       self.individual[1][1][0, 0].item(), self.individual[1][4][0, 0].item(),
		                       self.individual[1][23][0, 0].item(), self.individual[1][30][0, 0].item()])
		self.bestfits.append(self.best_solutions[1])
		self.avgfits.append(avg_fit)
		return self.best_solutions[1]


class PNESTrainer:
	def __init__(self, inp_size=50, population_size=5, time_budget=500, task_desc="", cpus=30, init_eta_m=10,
	             init_eta_s=2):
		"""指针网络的演化训练器

		Args:
			inp_size (int, optional): 训练数据的节点规模. Defaults to 50.
			population_size (int, optional): 种群数量. Defaults to 5.
			time_budget (int, optional): 训练时间预算. Defaults to 500.
			task_desc (str, optional): 任务描述（用于生成保存模型的文件夹）. Defaults to "".
		"""
		
		self.population_size = population_size  # population_size
		self.sampling_size = 15
		self.best_size = 5
		self.time_budget = time_budget * 60
		self.inp_size = inp_size
		self.cpus = min(cpus, self.population_size * self.sampling_size, os.cpu_count())
		self.forward_worker_num = self.sampling_size * self.population_size
		
		# hyperparameters
		self.batch_change_sigma = 10
		self.epoches = 5
		self.t_max = 10000
		self.batch_size = 256
		self.emb_size = 128
		self.hidden_size = 512
		self.r = 0.99
		self.nof_lstms = 5
		self.train_size = 1000
		self.test_size = 100
		self.sigma = 0.08
		self.init_eta = (init_eta_m, init_eta_s)
		
		self.model_hyper_param = {'embedding_dim': self.emb_size,
		                          'hidden_dim': self.hidden_size,
		                          'lstm_layers': self.nof_lstms,
		                          'dropout': 0.,
		                          'bidir': False}
		self.search_hyper_param = {'sampling_size': self.sampling_size,
		                           'init_eta': self.init_eta}
		
		# init normalized fitness by ranking
		self.normalized_fitnesses = [max(math.log(self.sampling_size / 2 + 1) - math.log(i + 1), 0)
		                             for i in range(self.sampling_size)]
		temp_sum = sum(self.normalized_fitnesses)
		self.normalized_fitnesses = [i / temp_sum - 1 / self.sampling_size for i in self.normalized_fitnesses]
		
		# init distribution population
		self.parameter_shapes = self.get_model_parameter_shapes()
		self.population = tuple(self.init_individual() for _ in range(self.population_size))
		self.best_solutions = []
		
		# if not task_desc: input(task_desc)
		# self.population = obj2id_list(self.population, self.temp_dir)
		
		self.folder = './res/task_desc_size-%d_pop-%d/' % (self.inp_size, self.population_size)
		os.makedirs(self.folder, exist_ok=True)
		self.logfile = self.folder + 'log.txt'
	
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
		return PointerNet(self.emb_size, self.hidden_size, self.nof_lstms, dropout=0., bidir=False)
	
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
		stds = self.normal_sample_using_shapes_list(self.parameter_shapes, 0.08, 0)
		sigmas = [i ** 2 for i in stds]
		return means, sigmas
	
	def evaluate_individual(self, model_param, dataset):
		"""计算个体在给定batch上的fitness值
			:param dataset:
			:param model_param:
		"""
		model = self.get_model_instance()
		param_dict = {name: param for name, param in zip(self.parameter_shapes, model_param)}
		model.load_state_dict(param_dict)
		
		with torch.no_grad():
			o, p = model(dataset)
		
		return calculate_paths(p, dataset)
	
	def train(self):
		"""开始训练
		"""
		train_dataset = TSPDataset(102400, self.inp_size, solve=False, random_seed=0)
		train_dataloader = DataLoader(train_dataset,
		                              batch_size=256,  # 256
		                              shuffle=True,
		                              num_workers=1)
		global best_samples_fut
		self.start_time = time.time()
		time_used = 0
		t = 0
		t_max = len(train_dataloader) * self.epoches  # a little different
		t_current = 0
		over = False
		
		forward_workers = [ForwardWorker.remote(self.model_hyper_param) for _ in range(self.forward_worker_num)]
		
		worker_seeds = torch.randint(0, 65535, (self.population_size,)).tolist()
		search_workers = []
		# print(worker_seeds)
		for i, seed in enumerate(worker_seeds):
			actor = NESWorker.remote(i, seed, self.parameter_shapes, forward_workers,
			                         **self.search_hyper_param)
			search_workers.append(actor)
		
		for epoch in range(self.epoches):
			# iterator = tqdm(train_dataloader, unit='Batch')
			for i_batch, sample_batched in enumerate(train_dataloader):
				# iterator.set_description('Batch %i/%i' % (epoch+1, 50000))
				
				# time_used = time.time() - self.start_time
				# if time_used >= self.time_budget:
				# 	over = True
				# 	break
				s = time.time()
				progress = t_current / t_max
				t_current += 1
				# iterator.set_description(
				# 	"%i/%i batch - %d/%d sec: " % (epoch + 1, self.epoches, int(time_used), self.time_budget))
				train_batch = Variable(sample_batched['Points'])
				
				search_tasks = [search_worker.search.remote(train_batch, progress) for search_worker in search_workers]
				# print(search_tasks) # diff
				best_samples_fut = ray.get(search_tasks)  # 同步
				assert len(best_samples_fut) == self.population_size
				print(f"\n==={t_current}/{t_max}=== {time.time() - s}s")
				print(best_samples_fut)
			# avg_length = sum(i[1] for i in self.best_solutions) / len(self.best_solutions)
			# print("AVG_L:", avg_length)
			# iterator.set_postfix({'avg_length': avg_length})
			# logger("%i/%i batch - %d/%d sec: %.2f length" % (
			# 	epoch + 1, self.epoches, int(time_used), self.time_budget, avg_length), self.logfile)
			
			if over:
				break
			t += 1
		# break  # =======================
		
		ray.get([search_worker.draw.remote() for search_worker in search_workers])
		
		# return
		best_samples = best_samples_fut
	
	# for i, model in enumerate(best_samples):
	# 	print("Saving models: %d/%d" % (i, len(best_samples)))
	# 	self.save_models(model[0], '%d.pth' % i)
	
	def test_best(self):
		"""选出在测试集上表现最好的Policy
		"""
		iterator = tqdm(test_dataloader, unit="Batch")
		epoch = 0
		self.test_best_result = []
		for i_batch, sample_batched in enumerate(iterator):
			iterator.set_description("batch: %i" % (epoch + 1))
			test_batch = Variable(sample_batched['Points'])
			batch_res = [self.evaluate_individual(ind_i[0], test_batch) for ind_i in self.best_solutions]
			self.test_best_result.append(batch_res)
		ma = torch.tensor(self.test_best_result)
		avg_length = torch.mean(ma, dim=0)
		
		max_mean_index = torch.argmin(avg_length)
		logger("Index of Best Policy:%d with min average length:%.2f" % (
			max_mean_index, avg_length[max_mean_index].item()), self.logfile)


if __name__ == '__main__':
	task_name = input("task_name:")
	os.makedirs(f"../results/{task_name}/")
	os.chdir(f"../results/{task_name}/")
	print(os.getcwd())
	os.makedirs(f"logs/", exist_ok=True)
	# inp_size = int(input("input_size:"))# 100  # [5,10,15,20,50,100,500,1000]
	inp_size = 20
	torch.manual_seed(0)
	# 准备数据
	# train_dataset = TSPDataset(102400, inp_size, solve=False, random_seed=0)
	# train_dataloader = DataLoader(train_dataset,
	#                               batch_size=256,  # 256
	#                               shuffle=True,
	#                               num_workers=1)
	
	test_dataset = TSPDataset(2000, inp_size, solve=False, random_seed=0)
	test_dataloader = DataLoader(test_dataset,
	                             batch_size=200,
	                             shuffle=True,
	                             num_workers=1)
	# 参数
	# parser = argparse.ArgumentParser(description="ENS: NCS-PtrNet")
	# # Data
	# parser.add_argument('--train_size', default=1000, type=int, help='Training data size')
	# parser.add_argument('--val_size', default=100, type=int, help='Validation data size')
	# parser.add_argument('--test_size', default=100, type=int, help='Test data size')
	# parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
	# # Train
	# parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
	# parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
	# # TSP
	# parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
	# # Network
	# parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
	# parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
	# parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
	# parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
	# parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')
	
	# # for Test
	# # parser.add_argument('--train_size', default=2000, type=int, help='Training data size')
	# # parser.add_argument('--val_size', default=500, type=int, help='Validation data size')
	# # parser.add_argument('--test_size', default=500, type=int, help='Test data size')
	# # parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
	
	# params = parser.parse_args()
	ray.init()
	trainer = PNESTrainer(inp_size=inp_size,
	                      population_size=1,
	                      time_budget=500,
	                      task_desc='test')
	trainer.train()

# trainer.test_best()
