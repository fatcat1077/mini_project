import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# 定義轉換（包含 normalization）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 載入 MNIST 訓練資料
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 定義反向 normalization，用以將圖片轉回原始像素值（可選）
inv_normalize = transforms.Normalize(
    mean=[-0.1307/0.3081],
    std=[1/0.3081]
)

# 選取要顯示的圖片數量（例如顯示 5 張）
num_images = 5
fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

for i in range(num_images):
    # 取出圖片與標籤
    img, label = train_dataset[i]
    
    # 若要顯示反向 normalization 後的圖片，則使用 inv_normalize 轉換
    img_inv = inv_normalize(img)
    
    # 將 tensor 轉成 numpy 陣列，並 squeeze 掉多餘的維度（因 MNIST 是單通道）
    img_np = img_inv.numpy().squeeze()
    
    # 顯示圖片，設定為灰階
    axes[i].imshow(img_np, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
