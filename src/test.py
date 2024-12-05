import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import matplotlib.pyplot as plt



env = gym.make('PongNoFrameskip-v4')
env = AtariPreprocessing(env,
                         scale_obs=False,
                         terminal_on_life_loss=True,
                         )
env = FrameStack(env, num_stack=4)
n_actions = env.action_space.n
state_dim = env.observation_space.shape

# env.render()
test = env.reset()
for i in range(100):
    test = env.step(env.action_space.sample())[0]

print(test)



plt.imshow(test.__array__()[0,...])