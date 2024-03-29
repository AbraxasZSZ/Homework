import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# 将读入的图像进行标准化处理
transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

get_label = lambda x: x.name.split('.')[0]

class get_dataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = list(Path(root).glob('*.jpg'))
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        label = get_label(img)
        label = 1 if label == 'dog' else 0
        if self.transform:
            img = self.transform(Image.open(img))
        return img, torch.tensor(label, dtype=torch.int64)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 26 * 26, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_path = './train'
    dataset = get_dataset(train_path, transform=transforms)
    train_data, valid_data = random_split(dataset,
                                          lengths=[int(len(dataset) * 0.8), int(len(dataset) * 0.2)],
                                          generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64)

    model = CNN()
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    train_loss_list = []
    train_acc_list = []

    for epoch in range(epochs):
        print("Epoch {} / {}".format(epoch + 1, epochs))

        t_loss, t_corr = 0.0, 0.0

        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
            loss = loss_function(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * inputs.size(0)
            t_corr += torch.sum(preds.argmax(1) == labels)

        train_loss = t_loss / len(train_loader.dataset)
        train_acc = t_corr.cpu().numpy() / len(train_loader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print('Train Loss: {:.4f} Accuracy: {:.4f}%'.format(train_loss, train_acc * 100))

    v_loss, v_corr = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs)
            v_loss += loss.item() * inputs.size(0)
            v_corr += torch.sum(preds.argmax(1) == labels)

        print('Valid Loss: {:.4f} Accuracy: {:.4f}%'.format(v_loss / len(valid_loader.dataset),
                                                            (v_corr / len(valid_loader.dataset)) * 100))

    test_path = './test'
    # get dataset
    test_data = get_dataset(test_path, transform=transforms)
    # print(len(test_data))
    test_loader = DataLoader(test_data, batch_size=64)
    _loss, _corr = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            y = model(inputs)
            preds = y.argmax(1)
            _loss += loss.item() * inputs.size(0)
            _corr += torch.sum(preds == labels)

        print('Test Loss: {:.4f} Accuracy: {:.4f}%'.format(_loss / len(test_loader.dataset),
                                                           (_corr / len(test_loader.dataset)) * 100))
