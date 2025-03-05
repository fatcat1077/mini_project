import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 設定隨機種子以利重現
np.random.seed(10)

# 1. 載入 MNIST 資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 資料前處理：將 28x28 圖片展平成 784 維向量，並正規化至 [0, 1]
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# 3. 將標籤轉換為 one-hot 編碼
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 4. 建立多層感知器 (MLP) 模型
model = Sequential()
model.add(Dense(units=1000, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 5. 編譯模型，設定損失函數、優化器與評估指標
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 6. 訓練模型
history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    epochs=20,
                    batch_size=200,
                    verbose=2)

# 7. 評估模型在測試集上的準確率
scores = model.evaluate(x_test, y_test)
print("\nTest accuracy:", scores[1])
model.save('my_mlp_model.h5')