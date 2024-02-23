# used to send commands
import pyautogui
# used to process frames
import cv2
import numpy as np

# Used to visualize captured frames
from matplotlib import pyplot as plt
import time
# Used for the environment 
import gymnasium as gym
from gymnasium import spaces
from dino_web_simple_game_driver import DinoWebSimpleGameDriver
from dino_web_advanced_game_driver import DinoWebAdvancedGameDriver

class DinoWebEnv(gym.Env):
    """Custom Environment for Dino - Chrome game, that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, driver=1):
        super().__init__()
        # Setup spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)

        if driver == 1:
            self.driver = DinoWebSimpleGameDriver()
        elif driver == 2:
            self.driver = DinoWebAdvancedGameDriver()
    
    ##-------------------------------------##
    
    def step(self, action):
        # 0 - up, 1 - down, 2 - nothing
        action_map = {
            0: 'space',
            1: 'down', 
            2: 'no_op'
        }

        if action != 2:
            pyautogui.press(action_map[action])

        done = self.get_game_over() 
        observation = self.get_observation()
        
        reward = 0.1 if not done else -1
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    ##-------------------------------------##

    def render(self):
        # TODO. move the the driver class
        plt.imshow(np.array(self.cap.grab(self.game_location))[:,:,:3])
        plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    ##-------------------------------------##
    
    def reset(self, seed=None, options=None):
        # Click anywhere in the chrome window to reset the game
        time.sleep(0.5)
        pyautogui.click(x=150, y=150)
        pyautogui.press('space')

        info = {}

        return self.get_observation(), info

    ##-------------------------------------##

    def close(self):
        self.driver.close()

    ##-------------------------------------##

    def get_observation(self, visualize=False):

        return self.driver.get_game_state(visualize=visualize)
        # return np.reshape(resized, (1, 83, 100))
        
    ##-------------------------------------##

    def get_game_over(self, visualize=False):

        return self.driver.is_game_over(visualize=visualize)
    

if __name__ == "__main__":

    print("-START-")
    env = DinoWebEnv()

    # print(env.action_space.sample())
    # plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_BGR2RGB))
    # plt.show()

    env.get_game_score(True)
    
    # done, done_cap = env.get_game_over(True)
    # print(done)

    # env.render()
    # env.reset()

    # for episode in range(10):
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         obs, reward, done, info = env.step(env.action_space.sample())
    #         total_reward += reward
    #     print('Total Reward for episode {} is {}'.format(episode+1, total_reward))

    print("-STOP-")