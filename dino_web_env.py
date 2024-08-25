# used to process frames
import cv2
import numpy as np
# used to visualize captured frames
from matplotlib import pyplot as plt
# used for the environment
import gymnasium as gym
from gymnasium import spaces
from game_drivers.dino_web_base_game_driver import DinoGameDriver
from game_drivers.dino_web_simple_game_driver import DinoWebSimpleGameDriver
from game_drivers.dino_web_advanced_game_driver import DinoWebAdvancedGameDriver


class DinoWebEnv(gym.Env):
    """Custom Environment for Dino - Chrome game, that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, driver=1):
        super().__init__()

        self.observation_space = spaces.Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)

        if driver == DinoGameDriver.BASIC:
            self.driver = DinoWebSimpleGameDriver()
        elif driver == DinoGameDriver.ADVANCED:
            self.driver = DinoWebAdvancedGameDriver()

    ##-------------------------------------##

    def step(self, action):
        # 0 - up, 1 - down, 2 - nothing
        if action != 2:
            self.driver.press(action)

        done = self.get_game_over()
        observation = self.get_observation()
        game_points = self.get_game_points()

        reward = 0.1 if not done else -1
        info = {
            "game_points": game_points
        }
        truncated = False

        return observation, reward, done, truncated, info

    ##-------------------------------------##

    def render(self):
        print("..render..")
        # TODO. move the the driver class
        plt.imshow(np.array(self.cap.grab(self.game_location))[:,:,:3])
        plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    ##-------------------------------------##
    
    def reset(self, seed=None, options=None):
        self.driver.reset()

        info = {}

        return self.get_observation(), info

    ##-------------------------------------##

    def close(self):
        self.driver.close()

    ##-------------------------------------##

    def get_observation(self, visualize=False):
        return self.driver.get_game_state(visualize=visualize)

    ##-------------------------------------##

    def get_game_over(self, visualize=False):
        return self.driver.is_game_over(visualize=visualize)

    ##-------------------------------------##

    def get_game_points(self, visualize=False):
        return self.driver.get_game_points(visualize=visualize)


if __name__ == "__main__":

    print("-START-")
    env = DinoWebEnv()

    env.get_game_points(True)

    # for episode in range(10):
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         obs, reward, done, info = env.step(env.action_space.sample())
    #         total_reward += reward
    #     print('Total Reward for episode {} is {}'.format(episode+1, total_reward))

    print("-STOP-")
