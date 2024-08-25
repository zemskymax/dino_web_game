# used to send commands
import time
import pyautogui
# Used for optical character recognition
import pytesseract
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mss import mss


class DinoWebSimpleGameDriver:
    def __init__(self):
        self.cap = mss()
        self.game_location = {'top': 400, 'left': 50, 'width': 500, 'height': 300}
        self.done_location = {'top': 430, 'left': 475, 'width': 450, 'height': 50}
        self.score_location = {'top': 370, 'left': 1160, 'width': 160, 'height': 50}

    ##-------------------------------------##

    def get_game_state(self, visualize=False):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))

        # 300x500x3 - normal
        # 300x500 - gray
        # 83x100 - resized
        if visualize:
            plt.imshow(resized)
            plt.show()

        return np.reshape(resized, (1, 83, 100))

    ##-------------------------------------##

    def get_game_points(self, visualize=False):
        score_cap = np.array(self.cap.grab(self.score_location))[:, :, :3]

        if visualize:
            plt.imshow(score_cap)
            plt.show()

        res = pytesseract.image_to_string(score_cap, config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789', lang='eng').replace(" ", "")[:4]

        print(res)

        return 1

    ##-------------------------------------##

    def is_game_over(self, visualize=False):
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]

        if visualize:
            plt.imshow(done_cap)
            plt.show()

        done_strings = ['GAME', 'GAHE']

        res = pytesseract.image_to_string(done_cap, config='--oem 3 --psm 6', lang='eng').replace(" ", "")[:4]
        if res in done_strings:
            return True

        return False

    ##-------------------------------------##

    def close(self):
        plt.close('all')

    ##-------------------------------------##

    def press(self, action):
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        pyautogui.press(action_map[action])

    ##-------------------------------------##

    def reset(self):
        # Click anywhere in the chrome window to reset the game
        pyautogui.click(x=150, y=150)
        pyautogui.press('space')
        time.sleep(0.5)


if __name__ == "__main__":

    print("-START-")

    simple = DinoWebSimpleGameDriver()
    simple.get_game_points(visualize=True)

    print("-STOP-")
