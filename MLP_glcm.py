from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from scipy import signal as sg
import numpy as np
import cv2
import os

# === laws texture 계산 함수 ===
def laws_texture(gray_image):
    (rows, cols) = gray_image.shape[:2]
    smooth_kernel = (1 / 25) * np.ones((5, 5))
    gray_smooth = sg.convolve(gray_image, smooth_kernel, "same")
    gray_processed = np.abs(gray_image - gray_smooth)

    # Law's Texture filter
    filter_vectors = np.array([
            [1, 4, 6, 4, 1],        # L5
            [-1, -2, 0, 2, 1],      # E5
            [-1, 0, 2, 0, 1],       # S5
            [1, -4, 6, -4, 1],      # R5
        ])

    filters = []
    for i in range(4):
        for j in range(4):
            filters.append(
                np.matmul(filter_vectors[i][:].reshape(5, 1), filter_vectors[j][:].reshape(1, 5))
            )

    # Convolution 계산 결과를 저장할 conv_maps
    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')

    # 9+1개 중요한 texture map 계산
    texture_maps = list()
    texture_maps.append((conv_maps[:, :, 1] + conv_maps[:, :, 4]) // 2)     # L5E5 / E5L5
    texture_maps.append((conv_maps[:, :, 2] + conv_maps[:, :, 8]) // 2)     # L5S5 / S5L5
    texture_maps.append((conv_maps[:, :, 3] + conv_maps[:, :, 12]) // 2)    # L5R5 / R5L5
    texture_maps.append((conv_maps[:, :, 7] + conv_maps[:, :, 13]) // 2)    # E5R5 / R5E5
    texture_maps.append((conv_maps[:, :, 6] + conv_maps[:, :, 9]) // 2)     # E5S5 / S5E5
    texture_maps.append((conv_maps[:, :, 11] + conv_maps[:, :, 14]) // 2)   # S5R5 / R5S5
    texture_maps.append(conv_maps[:, :, 10])                                # S5S5
    texture_maps.append(conv_maps[:, :, 5])                                 # E5E5
    texture_maps.append(conv_maps[:, :, 15])                                # R5R5
    texture_maps.append(conv_maps[:, :, 0])                                 # L5L5 (use to norm TEM)

    # Law's texture energy 계산
    TEM = list()
    for i in range(9):
        TEM.append(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9]).sum())
    return TEM


# === 이미지 패치에서 특징 추출 ===
train_dir = './texture_data/train'
test_dir = './textire_data/test'
classes = ['brick', 'grass', 'ground']

X_train = []
Y_train = []

PATCH_SIZE = 30
np.random.seed(1234)
for idx, texture_name in enumerate(classes):
   image_dir = os.path.join(train_dir, texture_name)
   for image_name in os.listdir(image_dir):
       image = cv2.imread(os.path.join(image_dir, image_name))
       image_s = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)

       for _ in range(10):
           h = np.random.randint(100-PATCH_SIZE)
           w = np.random.randint(100-PATCH_SIZE)

           image_p = image_s[h:h+PATCH_SIZE, w:w+PATCH_SIZE]
           image_p_gray = cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY)
           #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
           glcm = greycomatrix(image_p_gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)
           X_train.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                            greycoprops(glcm, 'correlation')[0, 0]] + laws_texture(image_p_gray))
           Y_train.append(idx)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print('train data: ', X_train.shape)
print('train label: ', Y_train.shape)

X_test = []
Y_test = []

for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(test_dir, texture_name)
    for image_name in os.listdir(image_dir)
        image = cv2.imread(os.path.join(image_dir, image_name))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(image_p_gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)
        X_test.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                            greycoprops(glcm, 'correlation')[0, 0]] + laws_texture(image_p_gray))
        Y_test.append(idx)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('test data: ', X_test.shape)
print('test label: ', Y_test.shape)
