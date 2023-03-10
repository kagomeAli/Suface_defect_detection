import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from skimage.feature import local_binary_pattern


def crop_resize_center_dataset(origin_path, crop_path, resize_path=True, max_number=5000, crop_shape=(500, 500),
                               resize_shape=(256, 256)):
    floders = folder_list
    if not len(folder_list):
        floders = os.listdir(origin_path)
    for folder in floders:
        img_path2 = origin_path + folder
        crop_path2 = crop_path + folder
        if resize_path:
            resize_path2 = resize_path + folder
            check_folder(resize_path2)
        check_folder(crop_path2)

        cur_list = []
        cur_noise_list = []
        # 影像存檔檔名
        count = 1
        for filename in os.listdir(img_path2):
            if 'jp' in filename:  # or 'jpg' in filename):
                # 讀影像
                # print(img_path2 + '/' + filename)
                img = cv2.imread((img_path2 + '/' + filename), cv2.IMREAD_GRAYSCALE)
                cur_shape = img.shape
                x = int(cur_shape[0] / 2 - crop_shape[0] / 2)
                y = int(cur_shape[1] / 2 - crop_shape[1] / 2)
                # 擷取原始影像中800 x 640的部分
                crop_img = img[y:y + crop_shape[0], x:x + crop_shape[1]]
                cv2.imwrite((resize_path2 + '/' + filename), crop_img)
                count += 1
            if count > max_number:
                break

        np.save(f'{crop_path}/{folder}.npy', cur_list)
        # np.save(f'{crop_path}/{folder}_lbp_noise.npy', cur_noise_list)

def check_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def generatorDefect(gray_image, gen_type=1):
    detect_w, detect_h = gray_image.shape

    defect_data = copy.deepcopy(gray_image)
    defect_label = np.zeros(np.shape(defect_data), dtype=np.uint8)  # 返回与图像 img1 尺寸相同的全零数组
    defect_value = random.randrange(0, 255)

    if gen_type == 0:
        # ellipse
        # detection's width height
        detect_w = random.randrange(10, 100)
        detect_h = random.randrange(int(detect_w / 4), int(detect_w * 2))
        # position center's x,y
        detect_x, detect_y = random.randrange(0, detect_w), random.randrange(0, detect_h)
        # 水平開始向逆時針偏移
        cur_angle = random.randrange(0, 45)
        # 繪製角度由0~180度的
        strat_ellipse = random.randrange(0, 100)
        end_ellipse = random.choice([110, 270, 360])

        # 中心座標是(256, 256)，長軸短軸分別是(70, 30)，由水平開始向逆時針偏移15度，繪製角度由0~180度的實心橢圓
        cv2.ellipse(defect_data, (detect_x, detect_y), (detect_w, detect_h), cur_angle, strat_ellipse, end_ellipse,
                    (defect_value), -1)
        cv2.ellipse(defect_label, (detect_x, detect_y), (detect_w, detect_h), cur_angle, strat_ellipse, end_ellipse,
                    (1), -1)
        defect_label

    elif gen_type == 1:
        # 畫粗度為7的紅色線
        start_positon = (random.randrange(0, detect_w), random.randrange(0, detect_h))
        end_positon = (random.randrange(0, detect_w), random.randrange(0, detect_h))
        defect_width = random.randrange(5, int(w / 5))
        cv2.line(defect_data, start_positon, end_positon, (defect_value), defect_width)
        cv2.line(defect_label, start_positon, end_positon, (1), defect_width)

    elif gen_type == 2:
        # detection's width height
        detect_w = random.randrange(10, 100)
        detect_h = random.randrange(int(detect_w / 4), int(detect_w * 2))
        # position center's x,y
        detect_x, detect_y = random.randrange(0, detect_w), random.randrange(0, detect_h)
        defect_radio = random.randrange(10, 40)
        cv2.circle(defect_data, (detect_x, detect_y), defect_radio, (defect_value), -1)  # -1 表示实心
        cv2.circle(defect_label, (detect_x, detect_y), defect_radio, (1), -1)  # -1 表示实心

    # imgAddMask1 = cv2.add(defect_data, np.zeros(np.shape(defect_data), dtype=np.uint8), mask=defect_data)  # 提取圆形 ROI

    return defect_data, defect_label


def matchData(normal_dataset, defect_dataset):
    len_normal = len(normal_dataset)
    len_defect = len(defect_dataset)

    half_normal_len = int(len_normal / 2)

    # len, width, height, channel
    normal_data = np.concatenate(
        (normal_dataset[0:half_normal_len, :, :, :], normal_dataset[half_normal_len:, :, :, :]), axis=3)

    normal_label = np.ones(half_normal_len)

    defect_data = np.concatenate((normal_dataset[0:len_defect, :, :, :], defect_dataset[0:len_defect, :, :, :]), axis=3)

    defect_label = np.zeros(len_defect)

    return normal_data, normal_label, defect_data, defect_label


def readData(image_path, image_size=(256, 256)):
    all_normal_data = []
    all_defect_data = []
    all_defect_label = []

    file_list = os.listdir(image_path)
    for file_index, file_name in file_list:
        img = cv2.imread(image_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, image_size, interpolation=cv2.INTER_NEAREST)
        all_normal_data.append(img)
        
        
        cur_defect_data, cur_defect_label = gen_defect(img, file_index % 3)
        all_defect_data.append(cur_defect_data)
        all_defect_label.append(cur_defect_label)
        

    all_normal_data = np.array(all_normal_data).reshape((-1,) + image_size).astype("float32")
    all_defect_data = np.array(all_defect_data).reshape((-1,) + image_size).astype("float32")
    all_defect_label = np.array(all_defect_label).reshape((-1,) + image_size).astype("float32")

    return all_normal_data, all_defect_data, all_defect_label

def traditionalLbp(img_gray):
    img_lbp= local_binary_pattern(img_gray, 8, 1.0, 'ror')
    return img_lbp