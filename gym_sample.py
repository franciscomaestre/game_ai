import gym


env = gym.make("SpaceInvaders-v0")
env.reset()

total_reward = 0

for _ in range(2000):
    env.render()
    next_obs, reward, done, info =env.step(env.action_space.sample())
    print(reward)