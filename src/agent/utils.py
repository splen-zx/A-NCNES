import random
import time

import ray
import torch
import numpy as np

from src.env.utils import get_env
from src.nn import nn_utils


class RolloutWorker:
	def __init__(self, nn_model, model_param, env_name, frame_limit, envs_num=15, env_param=None):
		self.model = nn_model(**model_param)
		self.model.set_parameter_grad(grad=False)
		self.env_name = env_name
		self.env_param = env_param
		self.envs = [get_env(self.env_name, self.env_param) for _ in range(envs_num)]
		
		self.envs_num = envs_num
		self.frame_limit = frame_limit
	
	def rollout(self, model_param):
		with torch.no_grad():
			device = nn_utils.get_device()
			self.model.set_parameters(model_param)
			g_reward = 0
			step = 0
			max_step = 0
			envs = self.envs.copy()
			env_seed = [random.randint(0, 2147483647) for _ in range(self.envs_num)]
			
			states = [env.reset(seed=seed)[0] for env, seed in zip(envs, env_seed)]
			while max_step < self.frame_limit and len(states) > 0:
				obs = torch.tensor(np.array(states)).to(device).float() / 255
				action_prob = self.model(obs)
				actions = torch.argmax(action_prob, dim=-1)
				
				step_ended_env = []
				states = []
				for i, env_action in enumerate(zip(envs, actions)):
					env, action = env_action
					state, reward, terminated, truncated, info = env.step(action.item())
					g_reward += reward
					if terminated or truncated:
						step_ended_env.append(i)
					else:
						states.append(state)
				step += len(envs)
				for i in step_ended_env[::-1]:
					env = envs.pop(i)
					env.close()
				max_step += 1
		return g_reward / self.envs_num, step / self.envs_num


@ray.remote
class RemoteRolloutWorker(RolloutWorker):
	pass


@ray.remote
def parallel_rollout(nn_model, model_args, env_name, env_param, frame_limit, envs_num, solution):
	model = nn_model(**model_args)
	model.set_parameter_grad(grad=False)
	envs = [get_env(env_name, env_param) for _ in range(envs_num)]
	
	with torch.no_grad():
		device = nn_utils.get_device()
		model.set_parameters(solution)
		g_reward = 0
		step = 0
		max_step = 0
		envs = envs.copy()
		env_seed = [random.randint(0, 2147483647) for _ in range(envs_num)]
		
		states = [env.reset(seed=seed)[0] for env, seed in zip(envs, env_seed)]
		while max_step < frame_limit and len(states) > 0:
			obs = torch.tensor(np.array(states)).to(device).float() / 255
			action_prob = model(obs)
			actions = torch.argmax(action_prob, dim=-1)
			
			step_ended_env = []
			states = []
			for i, env_action in enumerate(zip(envs, actions)):
				env, action = env_action
				state, reward, terminated, truncated, info = env.step(action.item())
				g_reward += reward
				if terminated or truncated:
					step_ended_env.append(i)
				else:
					states.append(state)
			step += len(envs)
			for i in step_ended_env[::-1]:
				env = envs.pop(i)
				env.close()
				del env
			max_step += 1
	solution_future = ray.put(solution)
	return g_reward / envs_num, step / envs_num, solution_future


@ray.remote
def get_best_tested_solution(prev_future, testing_future, per_step=False):
	if prev_future is None:
		test_res = []
		best_score = None
		best_solution_future = None
	else:
		test_res, best_score, best_solution_future = ray.get(prev_future)
		
	reward, step, solution_future = ray.get(testing_future)
	if per_step:
		score = reward / step  # TODO: not the per_step cal method
	else:
		score = reward
	
	test_res.append(score)
	if (best_score is None) or (best_score <= score):
		best_score = score
		best_solution_future = solution_future
	
	return test_res, best_score, best_solution_future


if __name__ == "__main__":
	from src.nn.dqn import DQN_Atari
	
	r = RolloutWorker(DQN_Atari, {'num_actions': 3}, 'Freeway', 10000)
	s = time.time()
	r.rollout(None)
	print(time.time() - s)
