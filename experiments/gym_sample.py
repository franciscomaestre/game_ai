import gym


#env = gym.make("LunarLander-v2")
#env = gym.make("Breakout-v0")
#env = gym.make("SpaceInvaders-v0")
env = gym.make("Pong-v0")
#env = gym.make("BipedalWalker-v3")
env.reset()

total_reward = 0

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))

for _ in range(1000):
    env.render()
    next_obs, reward, done, info = env.step(env.action_space.sample())
    print(env.unwrapped._action_set)
    print(env.unwrapped.get_action_meanings())
