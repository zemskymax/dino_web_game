

from stable_baselines3.common.logger import configure
# Check Environment    
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from dino_web_env import DinoWebEnv
from dino_web_callback import DinoWebCallback
import time


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
SEED = 7
RANK = 1


if __name__ == "__main__":
    print("-BEGINNING-")

    callback = DinoWebCallback(check_freq=300, save_path=CHECKPOINT_DIR)

    env = DinoWebEnv()
    # env.seed(SEED + RANK)
    # set_global_seeds(SEED)
    env_checker.check_env(env)

    model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=700000, learning_starts=1000)
    
    print("--TRAIN START--")
    timeStart = time.time()

    # total_timesteps=100000
    model.learn(total_timesteps=100, callback=callback)
    
    timeEnd = time.time()

    print("--TRAIN STOPPED after {}--".format(timeEnd- timeStart))

    # model.load('train/best_model_so_far')

    print("--TEST START--")
    timeStart = time.time()

    for episode in range(5): 
        obs, info = env.reset()
        done = False
        score = 0

        while not done: 
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(int(action))

            time.sleep(0.01)

            score += reward
        print('Total Reward for episode {} is {}'.format(episode + 1, score))
        time.sleep(0.1)

    timeEnd = time.time()
    print("--TEST STOPPED after {}--".format(timeEnd- timeStart))

    print("-END-")
