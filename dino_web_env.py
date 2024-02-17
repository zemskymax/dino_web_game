# Used to capture the screen
from typing import List, Tuple
# used to send commands
import pyautogui
# used to process frames
import cv2
import numpy as np
# Used for optical character recognition
import pytesseract
# Used to visualize captured frames
from matplotlib import pyplot as plt
import time
# Used for the environment 
# from gym import Env
import gymnasium as gym
# from gym.spaces import Box, Discrete
from gymnasium import spaces
from mss import mss


class DinoWebEnv(gym.Env):
# class WebGame(Env):
    def __init__(self):
        super().__init__()
        # Setup spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)

        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 600, 'left': 0, 'width': 600, 'height': 500}
        self.done_location = {'top': 600, 'left': 900, 'width': 830, 'height': 70}
    
    ##-------------------------------------##
    
    def step(self, action):
        # 0 - up, 1 - down, 2 - nothing
        action_map = {
            0: 'space',
            1: 'down', 
            2: 'no_op'
        }
        if action !=2:
            pyautogui.press(action_map[action])

        done, done_cap = self.get_game_over() 
        observation = self.get_observation()
        
        reward = 1 
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    ##-------------------------------------##

    def render(self):
        plt.imshow(np.array(self.cap.grab(self.game_location))[:,:,:3])
        plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    ##-------------------------------------##
    
    def reset(self, seed=None, options=None):
        # Click anywhere in the chrome window to reset the game
        time.sleep(1)
        pyautogui.click(x=150, y=150)
        pyautogui.press('space')

        info = {}

        return self.get_observation(), info

    ##-------------------------------------##

    def close(self):
        # cv2.destroyAllWindows()
        plt.close('all')

    ##-------------------------------------##

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))
        # plt.imshow(resized)
        channel = np.reshape(resized, (1, 83, 100))
        return channel

    ##-------------------------------------##

    def get_game_over(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]

        # print(done_cap.shape)
        # plt.imshow(done_cap)

        done_strings = ['GAME', 'GAHE']
        done=False
        # if np.sum(done_cap) < 44300000:
        #     done = True
        done = False
        res = pytesseract.image_to_string(done_cap, config='--oem 3 --psm 6', lang='eng').replace(" ", "")[:4]
        if res in done_strings:
            done = True
        return done, done_cap



if __name__ == "__main__":

    print("-START-")
    env = DinoWebEnv()

    # print(env.action_space.sample())
    # plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_BGR2RGB))
    # plt.show()

    # done, done_cap = env.get_game_over()
    # print(done)

    # env.render()
    # env.reset()

    for episode in range(10):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            total_reward += reward
        print('Total Reward for episode {} is {}'.format(episode+1, total_reward))

    print("-STOP-")