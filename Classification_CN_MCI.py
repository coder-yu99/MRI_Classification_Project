import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# 定义文件夹路径
data_folder = '/data/data03/boyang/pythonProject/multimodel classification/Unique_id_classification'

def load_image_paths_and_labels(data_folder):
    image_paths = []
    labels = []
    class_folders = {'MCI': 1, 'CN': 0}  # 只保留AD和CN

    for class_name, label in class_folders.items():
        class_folder = os.path.join(data_folder, class_name)
        for filename in os.listdir(class_folder):
            if filename.endswith('.nii'):
                image_paths.append(os.path.join(class_folder, filename))
                labels.append(label)

    return image_paths, labels

class PETDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        image = np.expand_dims(image, axis=0)  # 增加通道维度
        label = self.labels[idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], torch.tensor(sample['label'], dtype=torch.long)

class ResizeTransform:
    def __init__(self, output_size):
        assert isinstance(output_size, (tuple, int))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.tensor(image, dtype=torch.float32)
        image = nn.functional.interpolate(image.unsqueeze(0), size=self.output_size, mode='trilinear', align_corners=False).squeeze(0)
        return {'image': image, 'label': label}

class CNN3D(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2)
        self.dropout2 = nn.Dropout(0.4)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32 * 12 * 12 * 12, 128)  # 根据输入尺寸调整
        self.fc2 = nn.Linear(128, 2)  # 输出类别数为2



    def forward(self, x):
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))

        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))

        x = self.dropout2(x)

        x = self.pool3(self.bn3(torch.relu(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_model-5-fold-mci-cn.pth'):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    early_stopping = EarlyStopping(patience=patience, path='best_model-5-fold-mci-cn.pth')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load('best_model-5-fold-mci-cn.pth'))
    return val_accuracy

def cross_validate(model_class, dataset, criterion, num_folds=5, num_epochs=50, patience=5):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{num_folds}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)

        model = model_class(dropout_rate=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        val_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience)
        fold_results.append(val_accuracy)

    avg_val_accuracy = np.mean(fold_results)
    print(f'Average Validation Accuracy: {avg_val_accuracy:.2f}%')

if __name__ == "__main__":
    # 加载数据
    image_paths, labels = load_image_paths_and_labels(data_folder)
    transform = transforms.Compose([
        ResizeTransform((96, 96, 96))
    ])
    dataset = PETDataset(image_paths, labels, transform=transform)

    criterion = nn.CrossEntropyLoss()

    cross_validate(CNN3D, dataset, criterion, num_folds=5, num_epochs=50, patience=5)
