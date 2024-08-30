import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import torch.nn.functional as F

# 自定义数据集类
class MRIDataset(Dataset):
    def __init__(self, csv_file, transform=None, crop_size=(100, 120, 100)):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.crop_size = crop_size

    def center_crop(self, img, output_size):
        c, h, w, d = img.shape
        new_h, new_w, new_d = output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        front = (d - new_d) // 2
        img = img[:, top:top + new_h, left:left + new_w, front:front + new_d]
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 如果 DX 为 "Dementia"，跳过此样本
        if row['DX'] == 'Dementia':
            return self.__getitem__((idx + 1) % len(self.data))  # 递归地获取下一个样本

        img_path = row['final_path']
        img = nib.load(img_path).get_fdata()
        img = np.expand_dims(img, axis=0)  # 添加通道维度
        img = self.center_crop(img, self.crop_size)
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        # 提取表格数据
        age = torch.tensor([row['AGE']], dtype=torch.float32)
        mmse = torch.tensor([row['MMSE']], dtype=torch.float32)
        apoe4 = torch.tensor([row['APOE4']], dtype=torch.float32)

        # 拼接所有表格特征
        tabular_data = torch.cat([age, mmse, apoe4])

        # 将标签 (DX) 映射为数值
        label_mapping = {'CN': 0, 'MCI': 1}  # 假设 'CN' 对应 0，'MCI' 对应 1
        label = torch.tensor(label_mapping[row['DX']], dtype=torch.long)

        return img, tabular_data, label


# CNN模块提取3D MRI特征
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(128 * 12 * 15 * 12, 256)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 15 * 12)  # 展平
        x = torch.relu(self.fc1(x))
        return x

# 多模态网络：结合CNN提取的特征和表格数据特征
class MultiModalNet(nn.Module):
    def __init__(self, cnn, num_tabular_features=3, num_classes=2):
        super(MultiModalNet, self).__init__()
        self.cnn = cnn
        self.fc_tabular = nn.Linear(num_tabular_features, 128)
        self.fc_combined = nn.Linear(256 + 128, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, img, tabular_data):
        cnn_features = self.cnn(img)
        tabular_features = torch.relu(self.fc_tabular(tabular_data))
        combined = torch.cat([cnn_features, tabular_features], dim=1)
        x = torch.relu(self.fc_combined(combined))
        x = self.fc_out(x)
        return x

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
csv_file = '/data/data03/boyang/pythonProject/AD-Classification/csv-file/final_adni.csv'  # 替换为CSV文件路径
dataset = MRIDataset(csv_file)

# 数据集划分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 初始化模型、损失函数和优化器
cnn_model = CNN3D()
model = MultiModalNet(cnn=cnn_model).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# 存储训练过程数据
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')

# 训练和验证
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for img, tabular_data, labels in train_bar:
        img, tabular_data, labels = img.to(device), tabular_data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(img, tabular_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_bar.set_postfix(loss=running_loss / len(train_loader), accuracy=100 * correct / total)

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
        for img, tabular_data, labels in val_bar:
            img, tabular_data, labels = img.to(device), tabular_data.to(device), labels.to(device)
            outputs = model(img, tabular_data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_bar.set_postfix(loss=val_loss / len(val_loader), accuracy=100 * correct / total)

    val_accuracy = 100 * correct / total
    val_loss = val_loss / len(val_loader)
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # 保存验证损失最低的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_weights_path = f"/data/data03/boyang/pythonProject/AD-Classification/model_weight/multimodalnet_{timestamp}.pth"
        torch.save(model.state_dict(), model_weights_path)
        print(f"Saved model with validation loss: {val_loss:.4f} at {model_weights_path}")

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

# 为整个图添加一个总标题
plt.suptitle('CN vs MCI MultiModalNet')

plt.show()
