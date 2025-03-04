import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader

#---- 模型定義 -----------------------------------------------------------------
class SimpNetDropoutMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpNetDropoutMNIST, self).__init__()
        # 第 1 組：conv1 -> BN -> ReLU -> Dropout -> conv1_0 -> BN -> ReLU -> Dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(p=0.2)

        self.conv1_0 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn1_0 = nn.BatchNorm2d(128)
        self.drop1_0 = nn.Dropout(p=0.2)

        # 第 2 組：conv2 -> BN -> ReLU -> Dropout -> conv2_1 -> BN -> Pool -> Dropout -> conv2_2 -> BN -> Dropout
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop2 = nn.Dropout(p=0.2)

        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28->14
        self.drop2_1 = nn.Dropout(p=0.2)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.drop2_2 = nn.Dropout(p=0.2)

        # 第 3 組：conv3 -> BN -> Dropout
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(p=0.2)

        # 第 4 組：conv4 -> Pool -> BN -> Dropout -> conv4_1 -> BN -> Dropout -> conv4_2 -> BN -> Dropout -> Pool
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)   # 14->7
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(p=0.2)

        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.drop4_1 = nn.Dropout(p=0.2)

        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.drop4_2 = nn.Dropout(p=0.2)
        self.pool4_2 = nn.MaxPool2d(kernel_size=2, stride=2) # 7->3 (會是3x3)

        # conv4_0
        self.conv4_0 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_0 = nn.BatchNorm2d(512)
        self.drop4_0 = nn.Dropout(p=0.2)

        # cccp4 -> cccp5 -> poolcp5 -> cccp6 (原 Network In Network 1x1 卷積)
        self.cccp4 = nn.Conv2d(512, 2048, kernel_size=1)
        self.drop4_3 = nn.Dropout(p=0.2)

        self.cccp5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.poolcp5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3->1 (3x3 -> 1x1)
        self.drop4_5 = nn.Dropout(p=0.2)

        self.cccp6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # 在 prototxt 裡它最後還有 poolcp6，但 1x1 再 pool(kernel=2) 會變成 0x0，
        # 因此這裡省略最後那層 Pooling，以免張量維度變成(0,0)。

        # 最終全連接輸出 10 類 (MNIST)
        self.fc = nn.Linear(256*1*1, num_classes)

    def forward(self, x):
        # 第 1 組
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn1_0(self.conv1_0(x)))
        x = self.drop1_0(x)

        # 第 2 組
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool2_1(x)
        x = self.drop2_1(x)
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2_2(x)

        # 第 3 組
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)

        # 第 4 組
        x = self.conv4(x)
        x = self.pool4(x)
        x = F.relu(self.bn4(x))
        x = self.drop4(x)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.drop4_1(x)
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.drop4_2(x)
        x = self.pool4_2(x)

        x = F.relu(self.bn4_0(self.conv4_0(x)))
        x = self.drop4_0(x)

        # cccp4 ~ cccp6
        x = F.relu(self.cccp4(x))
        x = self.drop4_3(x)

        x = F.relu(self.cccp5(x))
        x = self.poolcp5(x)  # shape: [N,256,1,1]
        x = self.drop4_5(x)

        x = F.relu(self.cccp6(x))
        # 這裡若再 pool，就會變成 0×0；故略過

        # Flatten
        x = x.view(x.size(0), -1)  # shape: [N,256]
        x = self.fc(x)
        return x

#---- 關鍵：修改後的 get_mnist_dataloaders，將「原圖 + 反相」一同訓練 ----
def get_mnist_dataloaders(batch_size=64):
    """
    回傳 train_loader, test_loader，訓練集包含「原圖」+「反相圖」兩倍資料。
    """
    # 1. 先定義「正常」的 Transform
    transform_normal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 2. 再定義「反相」的 Transform
    transform_inverted = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),  # 反相: 讓黑變白、白變黑
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 3. 建立兩個 MNIST 資料集 (train=True)，一個正常、一個反相
    train_set_normal = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_normal
    )
    train_set_inverted = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_inverted
    )

    # 4. 透過 ConcatDataset 合併，得到兩倍大小的訓練集
    train_set_combined = ConcatDataset([train_set_normal, train_set_inverted])

    # 5. 建立 DataLoader
    train_loader = DataLoader(train_set_combined, batch_size=batch_size, shuffle=True)

    # -- 測試集 還是只用原本的 normal transform 就好 --
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#---- 簡易訓練流程 -------------------------------------------------------------
def train_simpnet_mnist(epochs=1, lr=0.001, device='cuda'):
    # 1. 取得 dataloader
    train_loader, test_loader = get_mnist_dataloaders()

    # 2. 初始化網路與優化器
    model = SimpNetDropoutMNIST(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 3. 開始訓練
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "simpnet_mnist.pth")
        # 4. 測試
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    print(torch.cuda.is_available())
    train_simpnet_mnist(epochs=5, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu')
