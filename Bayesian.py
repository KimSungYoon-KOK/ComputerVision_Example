from sklearn.metrics import accuracy_score, confusion_matrix  # 정확도 계산, confusion matrix 계산 함수
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from scipy import signal as sg
import itertools  # confusion matrix 시각화 함수에서 사용
import numpy as np
import cv2
import os


# ========== laws texture 계산 함수 ==========
def laws_texture(gray_image):
    (rows, cols) = gray_image.shape[:2]
    smooth_kernel = (1 / 25) * np.ones((5, 5))
    gray_smooth = sg.convolve(gray_image, smooth_kernel, "same")
    gray_processed = np.abs(gray_image - gray_smooth)

    # Law's Texture filter
    filter_vectors = np.array(
        [
            [1, 4, 6, 4, 1],        # L5
            [-1, -2, 0, 2, 1],      # E5
            [-1, 0, 2, 0, 1],       # S5
            [1, -4, 6, -4, 1],      # R5
        ]
    )

    filters = []
    for i in range(4):
        for j in range(4):
            filters.append(
                np.matmul(filter_vectors[i][:].reshape(5, 1), filter_vectors[j][:].reshape(1, 5))
            )

    # Convolution 연산 및 convmap 조합
    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], "same")

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
        #tem = round(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9]).sum(), 4)
        TEM.append(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9]).sum())
    return TEM



# ========== 이미지 패치에서 특징 추출 ==========
train_dir = './texture_data/train'
test_dir = './texture_data/test'
classes  = ['brick', 'grass', 'ground', 'water', 'wood']

X_train = []
Y_train = []

PATCH_SIZE = 32     # 16 or 32
GLCM_I = 1
GLCM_J = 0

np.random.seed(1234)
for idx, texture_name in enumerate(classes):
    img_dir = os.path.join(train_dir, texture_name)
    for img_name in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, img_name))
        img_s = cv2.resize(img, (100,100), interpolation=cv2.INTER_LINEAR)

        
        #print(img_name)
        for i in range(10):
            h = np.random.randint(100-PATCH_SIZE)
            w = np.random.randint(100-PATCH_SIZE)

            img_p = img_s[h:h+PATCH_SIZE, w:w+PATCH_SIZE]
            img_p_gray = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
            #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            glcm = greycomatrix(img_p_gray, distances=[GLCM_I], angles=[GLCM_J], levels=256, symmetric=False, normed=True)
            #dissimilarity = round(greycoprops(glcm, 'dissimilarity')[0, 0], 4)
            #correlation = round(greycoprops(glcm, 'correlation')[0, 0], 4)
            X_train.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                            greycoprops(glcm, 'correlation')[0, 0] ] + 
                            laws_texture(img_p_gray))
            Y_train.append(idx)
            #print('Crop ', i+1, ':  ', X_train[-1])
            
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# ========== Test 이미지에서 특징 추출 ==========
X_test = []
Y_test = []

for idx, texture_name in enumerate(classes):
    img_dir = os.path.join(test_dir, texture_name)
    for img_name in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, img_name))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(img_gray, distances=[GLCM_I], angles=[GLCM_J], levels=256, symmetric=False, normed=True)

        X_test.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                            greycoprops(glcm, 'correlation')[0, 0]] + 
                            laws_texture(img_p_gray))
        Y_test.append(idx)
        print(img_name, '  ', X_test[-1])

X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('test data: ', X_test.shape)
print('test label: ', Y_test.shape)



# ========== Bayesian classifier ==========
priors = []
covariances = []
means = []

for i in range(len(classes)):
    x = X_train[Y_train == i]
    priors.append((len(x) / len(X_train)))
    means.append(np.mean(x, axis=0))
    covariances.append(np.cov(np.transpose(x), bias=True))

# ========== likelihood 계산 함수 ===========
def likelihood(x, prior, mean, cov):
    return -0.5*np.linalg.multi_dot([np.transpose(x-mean), np.linalg.inv(cov), (x-mean)]) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)

Y_pred = []
for i in range(len(X_test)):
    likelihoods = []
    for j in range(len(classes)):
        likelihoods.append(likelihood(X_test[i], priors[j], means[j], covariances[j]))
    Y_pred.append(likelihoods.index(max(likelihoods)))
acc = accuracy_score(Y_test, Y_pred)
print('accuracy: ', acc)


    