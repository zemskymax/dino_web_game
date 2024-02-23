import base64
import numpy as np
import cv2
import os
import time
from io import BytesIO
from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from matplotlib import pyplot as plt


LEADING_TEXT = "data:image/png;base64,"


class DinoWebAdvancedGameDriver:
    def __init__(self):
        # chrome_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver")
        # chrome_path: str = 'chromedriver'

        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')
        options.add_argument("--mute-audio")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("disable-infobars")

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        try:
            self.driver.get('chrome://dino')
        except WebDriverException as e:
            print(e)
            # TODO. Raise an exception

        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas")))
                
    ##-------------------------------------##

    def get_game_state(self, visualize=False):
        img_str = self.driver.execute_script("return document.querySelector('canvas.runner-canvas').toDataURL()")
        img_str = img_str[len(LEADING_TEXT):]

        image_bytes = base64.b64decode(img_str)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8) # 3173
        raw = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        copped_raw = raw[:300, :500]
        gray = cv2.cvtColor(copped_raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))

        # 150x600x3 - normal
        # 150x600 - gray
        # 83x100 - resized
        if visualize:
            plt.imshow(resized)
            plt.show()
            # cv2.imwrite('decoded_image.png', gray)

        return np.reshape(resized, (1, 83, 100))
    
    ##-------------------------------------##
    
    def get_game_score(self, visualize=False):
        score = int(''.join(self.driver.execute_script("return Runner.instance_.distanceMeter.digits")))

        if visualize:
            print(score)

        return score

    ##-------------------------------------##

    def is_game_over(self, visualize=False):
        is_over = self.driver.execute_script("return Runner.instance_.crashed")

        if visualize:
            print(f'Is game over: {is_over}')

        return is_over

    ##-------------------------------------##

    def close(self):
        plt.close('all')
        self.driver.close()


if __name__ == "__main__":

    print("-START-")
    
    advanced = DinoWebAdvancedGameDriver()

    advanced.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)
    time.sleep(2)

    advanced.get_game_state(visualize=True)

    print("-STOP-")