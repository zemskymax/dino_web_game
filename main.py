import os 
import time

# from stable_baselines3.common.logger import configure
# Check Environment    
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from dino_web_env import DinoWebEnv
from dino_web_callback import DinoWebCallback
from datetime import timedelta
from game_drivers.dino_web_base_game_driver import DinoGameDriver


MODEL_DIR = './model/'
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
SEED = 7
RANK = 1


if __name__ == "__main__":
    print("-BEGINNING-")

    callback = DinoWebCallback(check_freq=1000, save_path=CHECKPOINT_DIR, driver_type=DinoGameDriver.ADVANCED)

    env = DinoWebEnv(driver=DinoGameDriver.ADVANCED)
    # env.seed(SEED + RANK)
    # set_global_seeds(SEED)
    env = Monitor(env)
    # env_checker.check_env(env)

    if os.path.exists(MODEL_DIR) and not os.path.isfile(MODEL_DIR) and os.listdir(MODEL_DIR):
        print("--LOAD--")
        model_names = os.listdir(MODEL_DIR)
        last_model_name = model_names[len(model_names) - 1]
        last_model_path = MODEL_DIR + last_model_name

        print("Loading {} mode.".format(last_model_path))

        model = DQN.load(last_model_path, env=env, tensorboard_log=LOG_DIR, device="cuda")
    else:
        print("--CREATE--")
        model = DQN('CnnPolicy', env=env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=700000, learning_starts=1000, device="cuda")
    
    print("--TRAIN START--")
    timeStart = time.time()

    model.learn(total_timesteps=20000, callback=callback, reset_num_timesteps=False, progress_bar=True)

    timeEnd = time.time()
    print("--TRAIN STOPPED after {}--".format(str(timedelta(seconds=(timeEnd - timeStart)))))

    time.sleep(5)

    print("--TEST START--")
    timeStart = time.time()

    for episode in range(5): 
        obs, info = env.reset()
        done = False
        score = 0

        while not done: 
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(int(action))

            score += reward

            time.sleep(0.01)

        print('Total Reward for episode {} is {}'.format(episode + 1, score))
        time.sleep(2)

    timeEnd = time.time()

    print("--TEST STOPPED after {}--".format(str(timedelta(seconds=(timeEnd - timeStart)))))

    env.close()

    print("--SAVE--")
    if os.path.exists(CHECKPOINT_DIR) and not os.path.isfile(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
        model_names = os.listdir(CHECKPOINT_DIR)
        model_names = [os.path.join(CHECKPOINT_DIR, f) for f in model_names]
        model_names.sort(key=os.path.getmtime)

        last_model_name = model_names[len(model_names) - 1]

        os.makedirs(MODEL_DIR, exist_ok=True)
        os.system(f'cp {last_model_name} {MODEL_DIR}')

    print("-END-")
