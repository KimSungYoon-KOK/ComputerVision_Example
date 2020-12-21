import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# === 신경망에 필요한 모듈 ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

# ================= 이미지 패치에서 특징 추출 =======================
train_dir = './archive/seg_train'
test_dir = './archive/seg_test'
classes  = ['buildings', 'forest', 'mountain', 'sea']

X_train = []
Y_train = []

np.random.seed(1234)
for idx, texture_name in enumerate(classes):
   image_dir = os.path.join(train_dir, texture_name)
   for image_name in os.listdir(image_dir):
       image = cv2.imread(os.path.join(image_dir, image_name))
       image_s = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)
       X_train.append(image_s)
       Y_train.append(idx)

X_train = np.array(X_train)/128 - 1
X_train = np.swapaxes(X_train, 1, 3)            # (Batch Size, input channel 개수, H, W)
Y_train = np.array(Y_train)
print('train data: ', X_train.shape)
print('train label: ', Y_train.shape)

X_test = []
Y_test = []

for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(test_dir, texture_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image_s = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)
        X_test.append(image_s)
        Y_test.append(idx)

X_test = np.array(X_test)/128 - 1
X_test = np.swapaxes(X_test, 1, 3)
Y_test = np.array(Y_test)
print('test data: ', X_test.shape)
print('test label: ', Y_test.shape)

# ================= 데이터셋 클래스 ======================
class Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        sample = (image, label)

        return sample

# === 신경망 모델 클래스 ===
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=3, kernel_size=3)
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=4)
        self.fc1 = nn.Linear(48, 4)
        self.relu = nn.ReLU6()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 10
learning_rate = 0.001
n_epoch = 100

Train_data = Dataset(images=X_train, labels=Y_train)
Test_data = Dataset(images=X_test, labels=Y_test)

Trainloader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)
Testloader = DataLoader(Test_data, batch_size=batch_size)

net = CNN()
net.to(device)
summary(net, (3, 150, 150), device='cuda' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
test_losses = []
test_accs = []


# ========== 학습 ==========
for epoch in range(n_epoch):
    train_loss = 0.0
    evaluation = []
    for i, data in enumerate(Trainloader, 0):
        features, labels = data
        labels = labels.long().to(device)
        features = features.to(device)
        optimizer.zero_grad()

        outputs = net(features.to(torch.float))
        _, predicted = torch.max(outputs.cpu().data, 1)
        evaluation.append((predicted==labels.cpu()).tolist())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss/(i+1)
    evaluation = [item for sublist in evaluation for item in sublist]
    train_acc = sum(evaluation)/len(evaluation)

    train_losses.append(train_loss)
    train_accs.append(train_acc)


    # === 테스트 ===
    if(epoch+1)%1 == 0:
        test_loss = 0.0
        evaluation = []
        for i, data in enumerate(Testloader, 0):
            features, labels = data
            labels = labels.long().to(device)
            features = features.to(device)

            outputs = net(features.to(torch.float))
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted==labels.cpu()).tolist())
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        
        test_loss = test_loss/(i+1)
        evaluation = [item for sublist in evaluation for item in sublist]
        test_acc = sum(evaluation)/len(evaluation)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print('[%3d/%d]\tloss: %.4f\tAccuracy : %.4f\t\tval-loss: %.4f\tval-Accuracy: %.4f' %
                (n_epoch, epoch+1, train_loss, train_acc, test_loss, test_acc))


# ===== 학습/테스트 loss/정확도 시각화 =====
plt.plot(range(len(train_losses)), train_losses, label='train loss')
plt.plot(range(len(test_losses)), test_losses, label='test loss')
plt.legend()
plt.show()
plt.plot(range(len(train_accs)), train_accs, label='train acc')
plt.plot(range(len(test_accs)), test_accs, label='test acc')
plt.legend()
plt.show()