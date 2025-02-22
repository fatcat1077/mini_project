import torch

# 查看是否偵測到 CUDA
print("CUDA is available:", torch.cuda.is_available())

# 如果有多張 GPU，也可以列出可用的 GPU 數
print("Number of GPUs:", torch.cuda.device_count())

# 查看目前 PyTorch 預設使用哪個 GPU（若有多張顯卡）
print("Current device:", torch.cuda.current_device())

# 如果想做簡單運算測試
x = torch.rand(1000, 1000, device='cuda')  # 建立在 GPU 上的張量
y = torch.rand(1000, 1000, device='cuda')
z = torch.matmul(x, y)  # 在 GPU 進行矩陣乘法
print("Matrix multiplication result shape:", z.shape)
