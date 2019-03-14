import numpy as np

class System():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0

        self.max_x = 2
        self.max_y = 2
        self.max_theta = np.pi/4    

    def reset(self):
        # sample random target state
        self.target_delta_x = np.random.uniform(low=-self.max_x, high=self.max_x)
        self.target_delta_y = np.random.uniform(low=-self.max_y, high=self.max_y)
        self.target_delta_theta = np.random.uniform(low=-self.max_theta, high=self.max_theta)
        self.target = np.array([self.target_delta_x, self.target_delta_y, self.target_delta_theta])
        return np.array([self.x, self.y, self.theta])

    def step(self, action):
        delta_x, delta_y, delta_theta = action
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        reward = np.exp(-0.5*np.sum(np.square(self.target-action)))  # squared distance
        done = False
        return np.array([self.x, self.y, self.theta]), reward, done

    def render(self):
        pass

def random_policy():
    x = np.random.uniform(low=-2, high=2)
    y = np.random.uniform(low=-2, high=2)
    theta = np.random.uniform(low=-np.pi/4 , high=np.pi/4 )
    return np.array([x, y, theta])

def debug():
    env = System()
    state = env.reset()
    for t in range(10):
        action = random_policy()
        next_state, reward, done = env.step(action)
        print('state {}\taction {}\tnext state {}\treward {}\tdone {}'.format(state, action, next_state, reward, done))

if __name__ == '__main__':
    debug()