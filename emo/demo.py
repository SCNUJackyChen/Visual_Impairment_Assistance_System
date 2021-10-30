from __future__ import print_function, division

import time
import wave

import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
import cv2
import librosa
from threading import Thread

from torch.utils.data import Dataset

emo_dict = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprised"}

class MyDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1 # datasets should be sorted!
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x2 = self.dataset2[index]
        x1 = self.dataset1[index]
        return x1, x2

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2)) # assuming both datasets have same length


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.img_conv1 = nn.Conv2d(3, 16, 5)
        self.aud_conv1 = nn.Conv1d(1, 16, 3)
        self.drop1 = nn.Dropout(0.5)
        self.img_pool = nn.MaxPool2d(4, 4)
        self.aud_pool = nn.MaxPool1d(2)
        self.img_conv2 = nn.Conv2d(16, 32, 3)
        self.img_pool2 = nn.MaxPool2d(2, 2)
        self.aud_conv2 = nn.Conv1d(16, 32, 3)
        # self.drop2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(32 * (62 * 30 + 17), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)
        self.softmax = nn.Softmax()
#01-01-01-01-01-01-02
    def forward(self, x_img, x_aud):
        x_img = self.img_pool(F.relu(self.img_conv1(x_img)))
        x_img = self.img_pool2(F.relu(self.img_conv2(x_img)))
        x_aud = self.aud_pool(F.relu(self.aud_conv1(x_aud)))
        x_aud = F.relu(self.aud_conv2(x_aud))
        # Note that simple concatination in this manner might not be the
        # best thing to do since one of the features might dominate the
        # other one. Hence, we can do
        #   1) Try with equal concatination
        #   2) Audio dominant
        #   3) Video dominant
        #   4) Make the values a hyperparameter
        #   5) Formulate a method to learn this composition ratio
        x_img = x_img.view(-1, 32 * 30 * 62)
        x_aud = x_aud.view(-1, 32 * 17)
        x = torch.cat([x_img, x_aud], dim=1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

trans = tv.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def test_net(vid,aud):
    net = torch.load("./emo/joint_cat.pth", map_location='cpu')
    # net.load_state_dict(torch.load("./model/net.pth"))
    if torch.cuda.is_available():
        net.to("cuda")
    net.eval()

    if torch.cuda.is_available():
        outputs = net(torch.FloatTensor(vid).to("cuda"),torch.FloatTensor(aud).to("cuda"))
    else:
        outputs = net(torch.FloatTensor(vid),torch.FloatTensor(aud))
    print(outputs)
    _, pred = torch.max(outputs, axis=1)
    print(emo_dict[pred.item()])

def preprocess(audio_path,vid_path):
    #audio
    X, sample_rate = librosa.load(audio_path+"/"+"output.wav", res_type='kaiser_best')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    mfccs = np.asarray(mfccs)
    mfccs = np.expand_dims(np.expand_dims(mfccs,axis=0),axis = 0)


    #video
    vidcap = cv2.VideoCapture(vid_path+"/"+"output.mp4")
    images = []
    success, image = vidcap.read()
    count = 0

    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 250))
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
        # image = image[150:662, 400:912]  # crop the image# [330,240,3]
        # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # resize the image
        images.append(image)
        success, image = vidcap.read()
        count += 1

    numPairs = len(images) // 2;
    for i in range(numPairs):
        collage = np.hstack([images[2 * i], images[2 * i + 1]])
        collage = np.transpose(collage, (2, 0, 1))
        collage = np.expand_dims(collage,axis=0)

    test_net(collage,mfccs)

def record(p,stream,i,CHUNK,FORMAT,CHANNELS,RECORD_SECONDS,RATE,audio_path):
    '''实现声音的录制
    '''
    # 打开数据流
    WAVE_OUTPUT_FILENAME = audio_path+"output"+".wav"

    print("* recording")

    # 开始录音
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording:",time.time())

    # 写入录音文件
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def capture(my_camera,images_path):
    #Start capture video
    sz = (int(my_camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(my_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    start = time.time()

    fps = 30
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    ## open and set props
    vout = cv2.VideoWriter()
    vout.open(images_path + 'output.mp4', fourcc, fps, sz, True)

    while time.time()-start <= 3:
        cnt = 0
        while cnt < fps:
            cnt += 1
            _, frame = my_camera.read()
            # cv2.putText(frame, str(cnt), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
            vout.write(frame)

    vout.release()

def realtime_audio():
    audio_path, vid_path = init()
    # 创建PyAudio对象
    # 定义数据流块
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    # 录音时间
    RECORD_SECONDS = 4

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    i = 0

    my_camera = cv2.VideoCapture(0)

    net = torch.load("./emo/joint_cat.pth", map_location = 'cpu')
    while True:
        tsk = []
        i+=1#Every two seconds
        print("start recorrd")
        t1 = Thread(target=record, args=(p, stream, i, CHUNK, FORMAT, CHANNELS, RECORD_SECONDS, RATE, audio_path,))
        t2 = Thread(target=capture, args=(my_camera, vid_path))
        tsk.append(t1)
        tsk.append(t2)
        t1.start()
        t2.start()
        for tt in tsk:
            tt.join()
        #==================================================
        # video_path = "./vid_input"
        # audio_path = "./aud_input"
        # test_flag = True
        log_dir = "./emo/joint_cat.pth"
        preprocess(audio_path, vid_path)

        break


def init():
    audio_path = "./emo/aud_input/"
    vid_path = "./emo/vid_input/"
    return audio_path, vid_path



if __name__ == "__main__":
    # aud_path = r"D:\NUS\IS\ITSS Project\speechemo\github\Emotion-recognition-using-audio-and-video-on-RAVDES-dataset-master\SUBMISSION\aud_input"
    # vid_path = r"D:\NUS\IS\ITSS Project\speechemo\github\Emotion-recognition-using-audio-and-video-on-RAVDES-dataset-master\SUBMISSION\vid_input"
    # preprocess(aud_path, vid_path)
    realtime_audio()
