import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条库

# 中心剪裁函数
def center_crop(img, output_size):
    c, h, w, d = img.shape
    new_h, new_w, new_d = output_size
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    front = (d - new_d) // 2
    img = img[:, top:top + new_h, left:left + new_w, front:front + new_d]
    return img

# 自定义数据集类
class MRIDataset(Dataset):
    def __init__(self, cn_dir, ad_dir, transform=None, crop_size=(100, 120, 100)):
        self.transform = transform
        self.crop_size = crop_size
        self.file_paths = []
        self.labels = []

        # 加载 CN 数据
        for filename in os.listdir(cn_dir):
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                self.file_paths.append(os.path.join(cn_dir, filename))
                self.labels.append(0)  # CN 标签为 0

        # 加载 AD 数据
        for filename in os.listdir(ad_dir):
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                self.file_paths.append(os.path.join(ad_dir, filename))
                self.labels.append(1)  # AD 标签为 1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        img = nib.load(img_path).get_fdata()
        img = np.expand_dims(img, axis=0)  # 添加通道维度

        # 中心剪裁
        img = center_crop(img, self.crop_size)
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, label

# CNN 模型
class MRI_CNN(nn.Module):
    def __init__(self):
        super(MRI_CNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # 假设输入图像大小为 100x120x100
        self.fc1 = nn.Linear(128 * 12 * 15 * 12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # 二分类，CN vs AD

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 15 * 12)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
cn_dir = '/data/data03/boyang/pythonProject/AD-Classification/gm-data/AD'  # 替换为 CN 文件夹路径
ad_dir = '//data/data03/boyang/pythonProject/AD-Classification/gm-data/CN'  # 替换为 AD 文件夹路径
dataset = MRIDataset(cn_dir, ad_dir)

# 数据集划分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 初始化模型、损失函数和优化器
model = MRI_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 存储训练过程数据
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 训练和验证
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_bar.set_postfix(loss=running_loss / (total / 8), accuracy=100 * correct / total)

    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(train_loader)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Validation", unit="batch")
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_bar.set_postfix(loss=val_loss / (total / 8), accuracy=100 * correct / total)

    val_accuracy = 100 * correct / total
    val_loss = val_loss / len(val_loader)
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

print("训练完成。")

# 绘制训练过程的损失和准确率图
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# 损失图
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 准确率图
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()
