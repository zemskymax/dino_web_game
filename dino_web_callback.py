# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
from game_drivers.dino_web_base_game_driver import DinoGameDriver


class DinoWebCallback(BaseCallback):

    def __init__(self, check_freq, save_path, driver_type=DinoGameDriver.BASIC, verbose=1):
        super(DinoWebCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.model_name = 'best_model_' + driver_type.name.lower()
    
    ##-------------------------------------##

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    ##-------------------------------------##

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, '{}_{}'.format(self.model_name, self.n_calls))
            self.model.save(model_path)

        return True

