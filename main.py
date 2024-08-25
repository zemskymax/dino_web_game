import argparse
import datetime
import glob
import os
import shutil
import time

from enum import Enum
# from stable_baselines3.common.logger import configure
# Check Environment    
from colorama import Fore
from stable_baselines3.common import env_checker
from stable_baselines3.common import utils
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from dino_web_custom_model import CustomActorCriticCnnPolicy
from dino_web_env import DinoWebEnv
from dino_web_callback import DinoWebCallback
from datetime import timedelta
from game_drivers.dino_web_base_game_driver import DinoGameDriver


MODEL_DIR = './model/'
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
SEED = 7
RANK = 1


class ModelAction(Enum):
    create = 'create'
    load = 'load'

    def __str__(self):
        return self.value

class ModelType(Enum):
    default = 'default'
    custom = 'custom'

    def __str__(self):
        return self.value

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--action", required=True,  type=ModelAction, choices=list(ModelAction), help='TODO')
    parser.add_argument('-mn', '--model_name', required=True, type=str, default="unnamed", help='TODO')
    parser.add_argument('-mt', '--model_type', required=True, type=ModelType, choices=list(ModelType), help='TODO')
    parser.add_argument('-t', '--test', required=False, type=bool, default=False, help='TODO')
    parser.add_argument('-d', '--debug', required=False, type=bool, default=False, help='TODO')
    parser.add_argument('-s', '--save', required=False, type=bool, default=False, help='TODO')

    return parser.parse_args()

def main():
    args = parse_args()

    print("-BEGINNING-")

    utils.set_random_seed(seed=SEED, using_cuda=True)
    print("Active device: " + str(utils.get_device("cuda")))

    model_name = str(args.model_name)
    print("Model name: " + model_name)

    env = DinoWebEnv(driver=DinoGameDriver.ADVANCED)
    # env = DummyVecEnv([lambda: env])
    env = Monitor(env)

    env_checker.check_env(env)

    if args.action == ModelAction.create:
        print("--CREATE ACTION--")
        logs_path = LOG_DIR  + model_name

        if args.model_type == ModelType.custom:
            print("--CUSTOM MODEL--")

            model = PPO(CustomActorCriticCnnPolicy, env=env, tensorboard_log=logs_path, verbose=1, device="cuda")
        elif args.model_type == ModelType.default:
            print("--DEFAULT MODEL--")

            # TODO. switch to PPO
            model = DQN('CnnPolicy', env=env, tensorboard_log=logs_path, verbose=1, buffer_size=700000, learning_starts=1000, device="cuda")
        else:
            print(Fore.RED + "--INVALID MODEL--")
            exit()

        time.sleep(1)

    elif args.action == ModelAction.load:
        print("--LOAD ACTION--")

        load_path = MODEL_DIR + model_name + "/"

        if os.path.exists(load_path) and not os.path.isfile(load_path):
            all_models = os.listdir(load_path)
            if len(all_models) >= 1:
                all_saved_models_path = [os.path.join(load_path, basename) for basename in all_models]
                last_saved_model_path = max(all_saved_models_path, key=os.path.getctime)

                print(f"Loading {last_saved_model_path} model from the {load_path} folder.")

                model = PPO.load(last_saved_model_path, env=env, tensorboard_log=LOG_DIR, device="cuda")
            else:
                print(Fore.RED + "Loading has failed!")
                exit()
    else:
        print(Fore.RED + "--INVALID ACTION--")
        exit()

    print("--TRAINING STARTED--")
    callback = DinoWebCallback(check_freq=1000, model_name=model_name, save_path=CHECKPOINT_DIR, driver_type=DinoGameDriver.ADVANCED)

    timeStart = time.time()
    model.learn(total_timesteps=10000, callback=callback, reset_num_timesteps=False, progress_bar=True)
    timeEnd = time.time()
    print("--TRAINING STOPPED after {}--".format(str(timedelta(seconds=(timeEnd - timeStart)))))

    if args.test:
        time.sleep(2)

        print("--TESTING STARTED--")
        best_score = 0
        timeStart = time.time()

        for episode in range(5):
            obs, info = env.reset()
            done = False
            score = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _truncated, info = env.step(int(action))
                score += reward
                time.sleep(0.01)

            if int(info["game_points"]) > best_score:
                best_score = int(info["game_points"])

            print('Total Reward for episode {} is {}'.format(episode + 1, score))
            time.sleep(2)

        timeEnd = time.time()

        print("--TESTING STOPPED after {}, best result: {}--".format(str(timedelta(seconds=(timeEnd - timeStart))), best_score))

    if args.debug:
        print("--DEBUG--")
        print(model.policy)

    if args.save:
        print("--SAVE--")
        if os.path.exists(CHECKPOINT_DIR) and not os.path.isfile(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
            models = os.listdir(CHECKPOINT_DIR)
            models_path = [os.path.join(CHECKPOINT_DIR, f) for f in models]
            # get the newest model
            models_path.sort(key=os.path.getmtime)
            last_model = models_path[len(models_path) - 1]
            # save the newest model
            os.makedirs(MODEL_DIR, exist_ok=True)
            # TODO. place in folder according to the algorithm
            curr_time = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            file_name = MODEL_DIR + model_name + "/" + curr_time + ".zip"
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            shutil.copy(last_model, file_name)
            # os.system(f'cp {last_model} {file_name}')

    print("--CLEANING--")
    env.close()

    if os.path.exists(CHECKPOINT_DIR) and not os.path.isfile(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)

    print("-END-")


if __name__ == "__main__":
    main()