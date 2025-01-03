import gymnasium as gym
import ale_py


def atari_env(name, karg=None):
	if karg is None:
		env = gym.make(f"{name}-v4", frameskip=1, render_mode="rgb_array")
	else:
		env = gym.make(f"{name}-v4", frameskip=1, render_mode="rgb_array", **karg)
	env = gym.wrappers.AtariPreprocessing(env)
	return gym.wrappers.FrameStackObservation(env, stack_size=4)


def freeway():
	return atari_env("Freeway")

def freeway_d():
	return atari_env("Freeway",
	                 {'difficulty': 1})

def beamrider():
	return atari_env("BeamRider")


def enduro():
	return atari_env("Enduro")


def breakout():
	return atari_env("Breakout")



config = {
	"Freeway_D": freeway_d,
	"Freeway": freeway,
	"Enduro": enduro,
	"BeamRider": beamrider,
	"Breakout": breakout
}


def get_env(env_name, env_kargs=None):
	if env_kargs is None:
		env_kargs = {}
	return config[env_name](**env_kargs)

if __name__ == '__main__':
	env = freeway()
	done = False
	step = 0
	env.reset()
	while not done:

		a = env.action_space.sample()
		s_, r, terminated, truncated, info = env.step(a)
		done = terminated or truncated
		s = s_
		step += 1
	print(step)
