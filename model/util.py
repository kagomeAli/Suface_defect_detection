
import numpy as np
import os, gc, cv2, shutil,platform

# 確認路徑資料夾是否存在 若無則創建
def check_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
