import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry",   1: "Disgusted", 2: "Fearful",  3: "Happy",
                4: "Neutral", 5: "Sad",       6: "Surprised"}

emoji_dist = {0: "./emojis/angry.png",   1: "./emojis/disgusted.png", 2: "./emojis/fearful.png",  3: "./emojis/happy.png",
              4: "./emojis/neutral.png", 5: "./emojis/sad.png",       6: "./emojis/surpriced.png"}

last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
show_text = [0]
cap1 = cv2.VideoCapture(0)
def show_vid():
    if not cap1.isOpened():
        print("cant open the camera1")
    # 参数ret为True或者False, 代表有没有读取到图片
    # 第二个参数frame表示截取到一帧的图片
    flag1, frame1 = cap1.read()
    # 要检测cascade文件是否在路径下，最后一般使用绝对路径
    bounding_box = cv2.CascadeClassifier("D:\\python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
    # 灰度图像
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        # 进行预测
        prediction = emotion_model.predict(cropped_img)
        # 识别的标签键值
        maxindex = int(np.argmax(prediction))
        print(maxindex)
        show_text[0] = maxindex
    if flag1 is None:
        print("Major error!")
    elif flag1:
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        frame2 = cv2.imread(emoji_dist[show_text[0]])
        img2 = Image.fromarray(frame2)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        lmain2.imgtk2 = imgtk2
        lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
        lmain2.configure(image=imgtk2)
        lmain.after(10, show_vid)


if __name__ == '__main__':
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open("background.png"))

    # 增加背景图片
    theLabel = tk.Label(root,justify = tk.LEFT,image = img,compound = tk.CENTER,fg = "white")
    theLabel.pack()

    # 设置摄像头窗口布局
    lmain = tk.Label(master=root, padx=50, bd=10)
    # 设置表情窗口布局
    lmain2 = tk.Label(master=root, bd=10)
    # 设置标签布局
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')

    lmain.pack(side=LEFT)
    lmain.place(x=50, y=180)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=800, y=180)
    lmain3.pack()
    lmain3.place(x=800, y=88)

    root.title("Photo To Emoji")
    root.geometry("1500x1000")

    show_vid()
    root.mainloop()

cap1.release()
cv2.destroyAllWindows()