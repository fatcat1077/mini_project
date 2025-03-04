# simpnet_mnist.py (範例名稱，可自行更改)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms


#---- 模型定義(與訓練時相同) ----
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

        # 第 2 組
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

        # 第 3 組
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(p=0.2)

        # 第 4 組
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
        self.pool4_2 = nn.MaxPool2d(kernel_size=2, stride=2) # 7->3

        self.conv4_0 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_0 = nn.BatchNorm2d(512)
        self.drop4_0 = nn.Dropout(p=0.2)

        self.cccp4 = nn.Conv2d(512, 2048, kernel_size=1)
        self.drop4_3 = nn.Dropout(p=0.2)

        self.cccp5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.poolcp5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3->1 (3x3->1x1)
        self.drop4_5 = nn.Dropout(p=0.2)

        self.cccp6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(256*1*1, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)

        x = F.relu(self.bn1_0(self.conv1_0(x)))
        x = self.drop1_0(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool2_1(x)
        x = self.drop2_1(x)

        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2_2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)

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

        x = F.relu(self.cccp4(x))
        x = self.drop4_3(x)

        x = F.relu(self.cccp5(x))
        x = self.poolcp5(x)
        x = self.drop4_5(x)

        x = F.relu(self.cccp6(x))
        x = x.view(x.size(0), -1)  # [N,256]
        x = self.fc(x)
        return x


#---- (B) 載入已訓練的模型並對 MNIST 測試集做推論 ----
def evaluate_model_on_mnist(model_path="simpnet_mnist.pth", batch_size=64, device='cpu'):
    """
    載入訓練好的 SimpNetDropoutMNIST，並在 MNIST 測試集上進行推論。
    預設 weights 檔名為 simpnet_mnist.pth
    """
    # 1. 建立模型 (結構要與訓練時相同)
    model = SimpNetDropoutMNIST(num_classes=10).to(device)

    # 2. 載入權重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 關閉 Dropout，BatchNorm 使用移動平均

    # 3. 準備 MNIST 測試集的資料載入
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                  download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 執行推論
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 前向傳播
            _, predicted = torch.max(outputs, 1)  # 取最大值之索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"MNIST Test Accuracy: {accuracy:.2f}%")

#---- (C) 如果只是想示範如何載入單張圖像 ----
def predict_single_image(model_path, image_path, device='cpu'):
    """
    對單張圖片做預測。
    需確保圖片能被轉換成 1通道 28x28 (MNIST 格式)。
    """
    import PIL
    from PIL import Image

    # 1. 建立模型
    model = SimpNetDropoutMNIST(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 讀取並前處理圖片
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # 轉成 1 通道
        transforms.Resize((28, 28)),                 # MNIST 大小
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img = Image.open(image_path)
    x = transform(img).unsqueeze(0).to(device)  # shape: [1,1,28,28]

    # 3. 前向傳播
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    print(f"此圖片預測結果是: {predicted_class}")


#---- (D) 測試程式入口 ----
if __name__ == "__main__":
    # 假設你訓練時也存了權重到 simpnet_mnist.pth
    # python simpnet_mnist.py

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # (1) 在 MNIST 測試集上測試
    #evaluate_model_on_mnist("simpnet_mnist.pth", batch_size=64, device=device)
    
    # (2) 若想測單張圖片 (路徑自行替換)
    predict_single_image("simpnet_mnist.pth", "test_photo1.png", device=device)
