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
    smooth_kernel = (1/25) * np.ones((5, 5))
    gray_smooth = sg.convolve(gray_image, smooth_kernel, "same")
    gray_processed = np.abs(gray_image - gray_smooth)

    # Law's Texture filter
    filter_vectors = np.array([
            [1, 4, 6, 4, 1],        # L5
            [-1, -2, 0, 2, 1],      # E5
            [-1, 0, 2, 0, 1],       # S5
            [1, -4, 6, -4, 1],      # R5
        ])                          # 16(4X4)개 filter를 저장할 filters

    filters = []
    for i in range(4):
        for j in range(4):
            filters.append(
                np.matmul(filter_vectors[i][:].reshape(5,1), filter_vectors[j][:].reshape(1,5))
            )

    # Convolution 연산 및 convmap 조합
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

    # Law's texture energy 계산: TEM 계산해서 L5L5로 정규화
    TEM = list()
    for i in range(9):
        TEM.append(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9]).sum())
    return TEM


# ========== 이미지 패치에서 특징 추출 ==========
train_dir = './archive/seg_train'
test_dir = './archive/seg_test'
classes  = ['buildings', 'forest', 'mountain', 'sea']

X_train = []
Y_train = []

PATCH_SIZE = 16
DISTANCE = 10
ANGLE = 0           #np.pi/2   np.pi/4
np.random.seed(1234)

print("Train Data에서 특징 추출")
for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(train_dir, texture_name)

    # 이미지 불러와서 100X100으로 축소
    for i, image_name in enumerate(os.listdir(image_dir)):
        image = cv2.imread(os.path.join(image_dir, image_name))
        w, h, _ = image.shape
        if w != 150 or h != 150:
            image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)
        
        print(f'9248/{(idx+1)*(i+1)}\r', end="")

        # 이미지에서 랜덤으로 10개의 패치를 잘라서 특징 추출
        for j in range(10):
            h = np.random.randint(150-PATCH_SIZE)
            w = np.random.randint(150-PATCH_SIZE)

            img_p = image[h:h+PATCH_SIZE, w:w+PATCH_SIZE]             # 패치 사이즈 만큼 이미지 크롭
            img_p_gray = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)        # 흑백 이미지로 변환
            #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            glcm = greycomatrix(img_p_gray, distances=[DISTANCE], angles=[ANGLE], levels=256, symmetric=False, normed=True)
            X_train.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                            greycoprops(glcm, 'correlation')[0, 0] ] +
                            laws_texture(img_p_gray))
            Y_train.append(idx)
            
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print('train data: ', X_train.shape)        # (92480, 11)  
print('train label: ', Y_train.shape)       # (92480)


# ========== Test 이미지에서 특징 추출 ==========
X_test = []
Y_test = []

print("Test Data에서 특징 추출")
for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(test_dir, texture_name)

    for i, image_name in enumerate(os.listdir(image_dir)):
        image = cv2.imread(os.path.join(image_dir, image_name))
        w, h, _ = image.shape
        if w != 150 or h != 150:
            image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                    # 흑백 이미지로 변환

        glcm = greycomatrix(image_gray, distances=[DISTANCE], angles=[ANGLE], levels=256, symmetric=False, normed=True)
        X_test.append([greycoprops(glcm, 'dissimilarity')[0, 0],
                            greycoprops(glcm, 'correlation')[0, 0]] + 
                            laws_texture(image_gray))
        Y_test.append(idx)
        print(f'1946/{(idx+1)*(i+1)}\r', end="")

X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('test data: ', X_test.shape)          # (1946, 11)
print('test label: ', Y_test.shape)         # (1946,)



# ========== Bayesian classifier ==========
priors = []
covariances = []
means = []

for i in range(len(classes)):                                   # 각 클래스마다
    X = X_train[Y_train == i]                                   # i번째 클래스 데이터를 X에 저장
    priors.append((len(X) / len(X_train)))                      # priors에 사전 확률 저장
    means.append(np.mean(X, axis=0))                            # mean에 평균값 저장
    covariances.append(np.cov(np.transpose(X), bias=True))      # covariances에 공분산 저장

# ========== likelihood 계산 함수 ==========
def likelihood(x, prior, mean, cov):
    return -0.5 * np.linalg.multi_dot([np.transpose(x-mean), np.linalg.inv(cov), (x-mean)]) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)

Y_pred = []                                                     # 예측 데이터 저장 list
for i in range(len(X_test)):                                    # 각 테스트 데이터에 대해
    likelihoods = []                   
    for j in range(len(classes)):                               # 모든 클래스의 likelihood 저장
        likelihoods.append(likelihood(X_test[i], priors[j], means[j], covariances[j]))
    Y_pred.append(likelihoods.index(max(likelihoods)))          # 가장 큰 likelihood를 채택

# 정확도 계산
acc = accuracy_score(Y_test, Y_pred)                           
print('accuracy: ', acc)


# ========== confusion matrix 시각화 ==========
def plot_confusion_matrix(cm, target_names=None, labels=True):
    accuracy = np.trace(cm) / float(np.sum(cm))

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    thresh = cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

        if labels:
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, "{:,}".format(cm[i,j]), horizontalalignment="center", color="white" if  cm[i,j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

# confusion matrix 시각화
plot_confusion_matrix(confusion_matrix(Y_test, Y_pred), target_names=classes)
