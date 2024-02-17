# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
# Check Environment    
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from dino_web_env import DinoWebEnv
import time


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
SEED = 7
RANK = 1


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    ##-------------------------------------##

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    ##-------------------------------------##

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True



if __name__ == "__main__":
    print("-START-")

    callback = TrainAndLoggingCallback(check_freq=300, save_path=CHECKPOINT_DIR)

    env = DinoWebEnv()
    # env.seed(SEED + RANK)
    # set_global_seeds(SEED)
    env_checker.check_env(env)

    # buffer_size=1200000
    model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=800000, learning_starts=1000)
    # total_timesteps=100000
    model.learn(total_timesteps=1000, callback=callback)

    # model.load('train/best_model_so_far')

    for episode in range(5): 
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _states = model.predict(obs)
            
            obs, reward, done, truncated, info = env.step(action)

            time.sleep(0.01)

            total_reward += reward
        print('Total Reward for episode {} is {}'.format(episode + 1, total_reward))
        time.sleep(1)

    print("-STOP-")
