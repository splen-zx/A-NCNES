import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import time
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import src.plt_utils


# wrapped_env = AtariPreprocessing(
# 	env=gym.envs.
# )

class AtariEnv:
	def __init__(self, env_name, env_param):
		env = gym.make(env_name, frameskip=1)

		self.warpped_env = AtariPreprocessing(env)
		
		


if __name__ == '__main__':

	
	ENV = "Freeway-v4"
	
	def display_frames_as_gif(frames, name):
		patch = plt.imshow(frames[0])
		plt.axis('off')
		
		def animate(i):
			patch.set_data(frames[i])
		
		anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1)
		anim.save(f'./breakout_result_{name}.gif', fps=30)
	
	
	def run_game(env, name, agent=None):
		
		print(env.observation_space)
		print(env.action_space)
		print(env.unwrapped.get_action_meanings())
		
		frames = []
		s, info = env.reset()
		time_s = time.time()
		step = 0
		tr = 0
		for i_episode in range(100000):
		
			# frames.append(s) #################
			# env.render()
			# time.sleep(0.1)
			if agent == None:
				a = env.action_space.sample()
			else:
				a = agent()
			
			s_, r, terminated, truncated, info = env.step(a)
			done = terminated or truncated
			s = s_
			tr+=r
			step +=1
			frames.append(env.render())
			if done:
				print(terminated)
				print(truncated)
				break
		print(step)
		print(tr)
		print((time.time()-time_s)/step*25000)
		
		env.close()
		
		src.plt_utils.display_frames_as_gif(frames, name)
		################
		# print(len(frames))
		# display_frames_as_gif(frames, name)
	
	# env = gym.make(ENV, frameskip=1, render_mode="rgb_array")
	# env = AtariPreprocessing(env)
	#
	# run_game(env, "warpped")
	
	env = gym.make(ENV, frameskip=1, render_mode="rgb_array")
	warpped_env = AtariPreprocessing(env.unwrapped)
	
	run_game(warpped_env, "un-warpped")
