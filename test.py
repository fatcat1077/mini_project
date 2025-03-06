from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
from PIL import Image

# 讀取已保存的模型
model = load_model('my_mlp_model.h5')

def predict_unknown_image(image_path):
    """
    讀取圖片、預處理並進行預測，返回預測的數字類別
    """
    # 1. 開啟圖片並轉為灰階
    img = Image.open(image_path).convert('L')
    
    # 2. 調整圖片大小為 28x28（MNIST 的尺寸）
    img = img.resize((28, 28))
    
    # 3. 轉換成 numpy 陣列
    img_array = np.array(img)
    
    # 如果圖片是白字黑底，而 MNIST 是黑字白底，可視情況反轉顏色：
    # img_array = 255 - img_array
    
    # 4. 將像素值正規化至 [0,1]
    img_array = img_array / 255.0
    
    # 5. 攤平成 1 維向量並增加 batch 維度，使其形狀符合 (1, 784)
    img_array = img_array.reshape(1, 784)
    
    # 6. 使用模型進行預測
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    return predicted_class[0]

# 範例：對一張未知圖片進行預測（請將 "unknown_image.png" 換成你的圖片檔案名稱）
for i in range(10):
    str_num = "%s" % i
    photos_path = Path("photos")
    image_path = photos_path / f"{i}.PNG"
    # 如果 predict_unknown_image() 函數需要字串，就轉成字串
    result = predict_unknown_image(str(image_path))
    print("預測結果",result)

