# 使用 CNN 进行面部情绪识别
# 作者：李腾腾
# 日期：2021-07-17
# =======================

# Imports
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 构建一个卷积神经网络架构，并在 FER2013 dataset  上训练模型，以便从图像中进行标签识别

# 初始化训练和验证生成器
train_dir = 'train'
val_dir = 'test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# 训练集处理
# target_size：可是实现对图片的尺寸转换，预处理
# color_mode：单通道图片：俗称灰度图，颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片
# batch_size：batch数据的大小
# "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
# 测试集处理
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# 构建卷积网络架构
emotion_model = Sequential()
# 第一部分卷积
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
# 第二部分卷积
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
# 池化+全连接
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

# 编译和训练模型
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# 指定优化器
optimizer = Adam(lr=0.0001, decay=1e-6)
# 指定loss函数-交叉熵损失函数
loss_func = 'categorical_crossentropy'
# accuracy真实标签和模型预测均为标量
emotion_model.compile(loss=loss_func,
                      optimizer=optimizer,metrics=['accuracy'])

# 对比fit，fit_generator节省内存
# steps_per_epoch：将一个epoch分为多少个steps，也就是划分一个batch_size多大，不能和batch_size共同使用
# epochs：训练的轮数epochs为50
# validation_steps：当steps_per_epoch被启用的时候才有用，验证集的batch_size
# validation_data：验证集
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# 保存模型权重至emotion_model.h5
emotion_model.save_weights('emotion_model.h5')

# 使用openCV haarcascade xml检测网络摄像头中人脸的边界框并预测情绪
# 启动摄像头
cap = cv2.VideoCapture(0)
while True:
    # 找到haar cascade在面周围绘制边界框
    ret, frame = cap.read()
    if not ret:
        break
    bounding_box = cv2.CascadeClassifier("D:\\python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 停止捕获视频和关闭相应的显示窗口的
cap.release()
cv2.destroyAllWindows()